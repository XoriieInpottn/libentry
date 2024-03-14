#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2023-09-06
"""

import argparse
import ast
import re
from dataclasses import is_dataclass, fields
from typing import Sequence, TypeVar, Type, Union, Optional

from .logging import logger

__all__ = [
    'literal_eval',
    'parse_all_args',
    'ArgumentParser',
    'inject_fields'
]


def literal_eval(exp):
    """Literal eval
    """
    try:
        return ast.literal_eval(exp)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return exp


def parse_all_args(known_args: argparse.Namespace, unknown_args: Sequence[str]):
    """Parse all args
    """
    name = None
    values = []
    for arg in unknown_args:
        if re.match(r'^--[a-zA-Z][\w\-]*$', arg):
            if name is not None:
                _fill_value(known_args, name, values)
                values = []
            name = arg[2:].replace('-', '_')
        elif arg.startswith('-'):
            raise argparse.ArgumentError(None, f'Invalid argument "{arg}".')
        else:
            values.append(arg)
    if name is not None:
        _fill_value(known_args, name, values)
    return known_args


def _fill_value(known_args, name, values):
    if len(values) == 0:
        raise argparse.ArgumentError(None, f'Lack of value for argument "{name}".')

    for i, value in enumerate(values):
        values[i] = literal_eval(value)

    known_args.__dict__[name] = values[0] if len(values) == 1 else values


T = TypeVar('T')


class ArgumentParser(argparse.ArgumentParser):
    PATTERN_ARG_NAME = re.compile(r"^--?[a-zA-Z][\w\-.]*$")
    PATTERN_ARG_PREFIX = re.compile(r"^--?")
    DATACLASS_OBJ_KEY = 'target'

    def __init__(self):
        super().__init__()
        self.obj_dict = {}

    def add_argument(self, *args, **kwargs):
        if self.DATACLASS_OBJ_KEY in kwargs:
            obj = kwargs[self.DATACLASS_OBJ_KEY]
            for prefix in args:
                self.obj_dict[prefix] = obj
            return obj
        elif len(args) >= 2 and is_dataclass(args[-1]) and all(isinstance(arg, str) for arg in args[:-1]):
            obj = args[-1]
            for prefix in args[:-1]:
                self.obj_dict[prefix] = obj
            return obj
        else:
            return super().add_argument(*args, **kwargs)

    def add_dataclass(self, prefix, d: Union[Type[T], T]) -> T:
        assert is_dataclass(d)
        if isinstance(d, type):
            d = d()
        self.obj_dict[prefix] = d
        return d

    def parse_args(self, args=None, namespace=None, parse_unknown=False):
        args, unknown_args = super().parse_known_args()

        d = {}
        name = None
        values = []
        for arg in unknown_args:
            if arg.startswith('-'):
                if self.PATTERN_ARG_NAME.match(arg):
                    if name is not None:
                        d[name] = values[0] if len(values) == 1 else values
                        values = []
                    name = self.PATTERN_ARG_PREFIX.sub("", arg).replace('-', '_')
                else:
                    value = literal_eval(arg)
                    if isinstance(value, str):
                        logger.warning(f'The value "{arg}" may be incorrect.')
                    values.append(value)
            else:
                values.append(literal_eval(arg))
        if name is not None:
            d[name] = values[0] if len(values) == 1 else values

        for name, value in d.items():
            sections = name.split('.')
            if len(sections) == 1:
                args.__dict__[name] = value
            else:
                prefix = sections[0]
                members = sections[1:-1]
                attr = sections[-1]
                if prefix not in self.obj_dict:
                    raise RuntimeError(f'There is no {prefix} argument.')
                obj = self.obj_dict[prefix]

                for member in members:
                    if not hasattr(obj, member):
                        raise RuntimeError(f'There is no {prefix} argument {name}.')
                    obj = getattr(obj, member)

                if not hasattr(obj, attr):
                    raise RuntimeError(f'There is no {prefix} argument {name}.')
                setattr(obj, sections[-1], value)

        return args


def inject_fields(dst, src, blacklist: Optional[Sequence[str]] = None):
    dst_field_names = {_field.name for _field in fields(dst)}
    for src_field in fields(src):
        name = src_field.name
        if blacklist and name in blacklist:
            continue
        if name in dst_field_names:
            value = getattr(src, name)
            setattr(dst, name, value)
    return dst
