#!/usr/bin/env python3

"""
@author: Guangyi
@since: 2023-09-06
"""

__all__ = [
    'literal_eval',
    'parse_all_args',
    'ArgumentParser',
    'inject_fields'
]

import argparse
import ast
import json
import re
from dataclasses import fields, is_dataclass
from typing import Dict, List, Optional, Sequence, Type, Union, get_args, get_origin

import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined


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


class DefaultValue:

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)


class ArgumentParser(argparse.ArgumentParser):

    def __init__(
            self,
            prog=None,
            usage=None,
            description=None,
            epilog=None,
            parents=(),
            formatter_class=argparse.HelpFormatter,
            prefix_chars='-',
            fromfile_prefix_chars=None,
            argument_default=None,
            conflict_handler='error',
            add_help=True,
            allow_abbrev=True
    ) -> None:
        super().__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            parents=parents,
            formatter_class=formatter_class,
            prefix_chars=prefix_chars,
            fromfile_prefix_chars=fromfile_prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler,
            add_help=add_help,
            allow_abbrev=allow_abbrev,
        )
        self.schema_dict = {}

    def add_schema(self, name: str, schema: Type[BaseModel], default: Union[str, BaseModel] = None):
        if default is not None:
            if isinstance(default, str):
                default_json = yaml.safe_load(default)
                if isinstance(default_json, str):
                    with open(default_json, "r") as f:
                        default_json = yaml.safe_load(f)
                if not isinstance(default_json, dict):
                    raise ValueError(f"Invalid default value for \"{name}\".")
            elif isinstance(default, BaseModel):
                default_json = default.model_dump()
            else:
                raise TypeError(f"Invalid default type {type(default)} for \"{name}\".")
            default_flat_dict = {}
            self._json_flatten(name, default_json, default_flat_dict)
            default = default_flat_dict

        self.add_argument(f"--{name}")
        if name in self.schema_dict:
            raise ValueError(f"Schema \"{name}\" exists.")
        self.schema_dict[name] = (schema, default)
        self._add_schema(name, schema)

    def _add_schema(self, prefix: str, schema: Type[BaseModel], instance: Optional[BaseModel] = None):
        model_fields = schema.model_fields
        assert isinstance(model_fields, dict)
        for name, info in model_fields.items():
            anno = info.annotation
            nested = False

            if isinstance(anno, type) and issubclass(anno, BaseModel):
                nested = True

            if get_origin(anno) is Union:
                # Here we just choose the first BaseModel.
                # So you should avoid define a Union contains multiple BaseModel types.
                for sub_anno in get_args(anno):
                    if isinstance(sub_anno, type) and issubclass(sub_anno, BaseModel):
                        anno = sub_anno
                        nested = True
                        break

            default_value = info.default
            if info.default_factory:
                default_value = info.default_factory()
            if isinstance(instance, schema):
                default_value = getattr(instance, name, None)

            if nested:
                self._add_schema(
                    f"{prefix}.{name}",
                    schema=anno,
                    instance=default_value
                )
            else:
                desc = info.description + " " if info.description else ""
                if info.is_required():
                    desc += "(required)"
                else:
                    if default_value is None:
                        desc += "(default=None)"
                    elif isinstance(default_value, (str, int, float, bool)):
                        desc += f"(default={default_value})"
                    elif isinstance(default_value, (Dict, List)):
                        try:
                            desc += f"(default={json.dumps(default_value)})"
                        except TypeError:
                            default_type = type(default_value).__name__
                            desc += f"(default={default_type} object)"
                    else:
                        default_type = type(default_value).__name__
                        desc += f"(default={default_type} object)"

                self.add_argument(
                    f"--{prefix}.{name}",
                    type=literal_eval,
                    default=DefaultValue(default_value),
                    help=desc
                )

    def parse_args(self, args=None, namespace=None):
        args = super().parse_args(args=None, namespace=None)

        config_flat_dict = {}
        for name, (_, default) in self.schema_dict.items():
            if default is not None:
                config_flat_dict.update(default)

            config_file = args.__dict__[name]
            if config_file is not None:
                with open(config_file, "r") as f:
                    config_json = yaml.load(f, yaml.FullLoader)
                    self._json_flatten(name, config_json, config_flat_dict)

        args_flat_dict = {
            name: value
            for name, value in args.__dict__.items()
            if "." in name
        }

        merged_flat_dict = {**config_flat_dict}
        for name, value in args_flat_dict.items():
            if name not in merged_flat_dict:
                if isinstance(value, DefaultValue):
                    value = value.value
                merged_flat_dict[name] = value
            else:
                if not isinstance(value, DefaultValue):
                    merged_flat_dict[name] = value
        merged_flat_dict = {
            k: v for k, v in merged_flat_dict.items()
            if v is not None and v is not PydanticUndefined
        }

        merged_json = {}
        self._json_unflatten(merged_flat_dict, merged_json)

        for name, (schema, _) in self.schema_dict.items():
            model = schema.model_validate(merged_json[name])
            args.__dict__[name] = model
        return args

    @staticmethod
    def _json_flatten(prefix: str, json: dict, output: dict):
        for name, value in json.items():
            if isinstance(value, dict):
                ArgumentParser._json_flatten(f"{prefix}.{name}", value, output)
            else:
                output[f"{prefix}.{name}"] = value

    @staticmethod
    def _json_unflatten(flat_dict: dict, output: dict):
        for name, value in flat_dict.items():
            path_list = name.split(".")
            sub_json = output
            for prefix in path_list[:-1]:
                if prefix not in sub_json:
                    sub_json[prefix] = {}
                sub_json = sub_json[prefix]
            sub_json[path_list[-1]] = value


def inject_fields(dst, src, blacklist: Optional[Sequence[str]] = None):
    dst_field_names = {*_list_field_names(dst)}
    for name in _list_field_names(src):
        if blacklist and name in blacklist:
            continue
        if name in dst_field_names:
            value = getattr(src, name)
            setattr(dst, name, value)
    return dst


def _list_field_names(obj):
    if is_dataclass(obj):
        return [_field.name for _field in fields(obj)]
    elif isinstance(obj, BaseModel):
        return [*obj.model_fields]
    else:
        return [obj.__dict__]
