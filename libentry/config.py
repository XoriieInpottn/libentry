#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2023-09-06
"""

import argparse
import ast
import inspect
import re
from typing import MutableMapping, Sequence, Mapping

__all__ = [
    'ConfigFromArguments',
    'literal_eval',
    'parse_all_args',
    'ArgumentParser'
]


class ConfigFromArguments(MutableMapping):
    """Config
    """
    empty = inspect.Parameter.empty

    def __init__(self, cls, obj=None, **kwargs):
        self.__cls = cls

        args = {}
        signature = inspect.signature(cls.__init__ if isinstance(cls, type) else cls)
        for name, param in signature.parameters.items():
            if isinstance(cls, type) and name == 'self':
                continue
            if param.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD:
                continue
            value = param.default
            ann = param.annotation
            if (ann is ConfigFromArguments.empty) and (value is not ConfigFromArguments.empty):
                ann = type(param.default)
            args[name] = [ConfigFromArguments.empty, ann, value]

        self.__args = args

        if obj is not None:
            self.load(obj)

        if kwargs:
            self.load(kwargs)

    def load(self, obj):
        """Load
        """
        if isinstance(obj, Mapping):
            for name in self:
                if name in obj:
                    setattr(self, name, obj[name])
        else:
            for name in self:
                if hasattr(obj, name):
                    setattr(self, name, getattr(obj, name))

    def build(self):
        """Build
        """
        args = {}
        for name, value in self.items():
            if value is ConfigFromArguments.empty:
                cls_name = self.__cls.__name__
                raise ValueError(f'"{name}" must be given to construct {cls_name}.')
            args[name] = value
        return self.__cls(**args)

    def get_type(self, name):
        """Get type
        """
        return self.__args[name][1]

    def __getattr__(self, name):
        item = self.__args[name]
        value = item[0]
        if value is ConfigFromArguments.empty:
            value = item[2]
        return value

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(ConfigFromArguments, self).__setattr__(name, value)
        else:
            self.__args[name][0] = value

    def __len__(self):
        return len(self.__args)

    def __iter__(self):
        return iter(self.__args)

    def __getitem__(self, name):
        item = self.__args[name]
        value = item[0]
        if value is ConfigFromArguments.empty:
            value = item[2]
        return value

    def __setitem__(self, name, value):
        self.__args[name][0] = value

    def __delitem__(self, name):
        del self.__args[name]

    def __str__(self):
        pairs = []
        max_len = 0
        for name, (value, _, _) in self.__args.items():
            if value is ConfigFromArguments.empty:
                continue

            if value is None:
                value = 'None'
            elif isinstance(value, str):
                value = f'\'{value}\''
            elif isinstance(value, (bool, int, float, list, tuple, dict)):
                value = str(value)
            else:
                type_name = value.__class__.__qualname__
                value = f'{type_name} instance at {id(value)}'

            break_line = value.find('\n')
            if break_line >= 0:
                value = value[:break_line] + '...'
            if len(value) > 64:
                value = value[:64] + '...'

            pairs.append((name, value))

            name_len = len(name)
            if name_len > max_len:
                max_len = name_len

        return '\n'.join([
            f'{name.rjust(max_len)}: {value}'
            for name, value in pairs
        ])

    def __repr__(self):
        pairs = []
        max_len = 0
        for name, (value, _, def_value) in self.__args.items():
            if value is ConfigFromArguments.empty:
                value = def_value

            if value is None:
                value = 'None'
            elif isinstance(value, str):
                value = f'\'{value}\''
            elif isinstance(value, (bool, int, float, list, tuple, dict)):
                value = str(value)
            else:
                type_name = value.__class__.__qualname__
                value = f'{type_name} instance at {id(value)}'

            break_line = value.find('\n')
            if break_line >= 0:
                value = value[:break_line] + '...'
            if len(value) > 64:
                value = value[:64] + '...'

            pairs.append((name, value))

            name_len = len(name)
            if name_len > max_len:
                max_len = name_len

        return '\n'.join([
            f'{name.rjust(max_len)}: {value}'
            for name, value in pairs
        ])


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


class ArgumentParser(argparse.ArgumentParser):
    """Argument parser
    """

    def parse_args(self, args=None, namespace=None, parse_unknown=False):
        return parse_all_args(*super(ArgumentParser, self).parse_known_args())
