#!/usr/bin/env python3

"""
@author: xi
"""

from dataclasses import dataclass, field
from inspect import signature, Parameter
from typing import *

__all__ = [
    'ConfigableCaller'
]


@dataclass
class _ArgumentItem:
    value: Any = field(default=Parameter.empty)
    type: Type = field(default=Parameter.empty)
    default: Any = field(default=Parameter.empty)


class ConfigableCaller(MutableMapping):

    def __init__(self, wrapped: Callable, obj=None, **kwargs):
        self._callable = wrapped

        self._args = {}
        sig = signature(wrapped.__init__ if isinstance(wrapped, type) else wrapped)
        for name, param in sig.parameters.items():
            if isinstance(wrapped, type) and name == 'self':
                continue
            if param.kind != Parameter.POSITIONAL_OR_KEYWORD:
                continue
            value = param.default
            ann = param.annotation
            if (ann is Parameter.empty) and (value is not Parameter.empty):
                ann = type(param.default)
            self._args[name] = _ArgumentItem(type=ann, default=value)

        if obj is not None:
            self.load(obj)
        if kwargs:
            self.load(kwargs)

    def get_value(self, name: str) -> Any:
        return self._args[name].value

    def get_type(self, name: str) -> Type:
        return self._args[name].type

    def get_default(self, name: str) -> Any:
        return self._args[name].default

    def load(self, obj):
        if isinstance(obj, Mapping):
            for name in self:
                if name in obj:
                    setattr(self, name, obj[name])
        else:
            for name in self:
                if hasattr(obj, name):
                    setattr(self, name, getattr(obj, name))

    def __call__(self):
        return self.build()

    def build(self):
        kwargs = {}
        for name, value in self.items():
            if value is Parameter.empty:
                raise ValueError(f'"{name}" must be given.')
            kwargs[name] = value
        return self._callable(**kwargs)

    def __getattr__(self, name):
        item = self._args[name]
        return item.default if item.value is Parameter.empty else item.value

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(ConfigableCaller, self).__setattr__(name, value)
        else:
            self._args[name].value = value

    def __len__(self):
        return len(self._args)

    def __iter__(self):
        return iter(self._args)

    def __getitem__(self, name):
        item = self._args[name]
        return item.default if item.value is Parameter.empty else item.value

    def __setitem__(self, name, value):
        self._args[name].value = value

    def __delitem__(self, name):
        del self._args[name]

    def __str__(self):
        pairs = []
        max_len = 0
        for name, item in self._args.items():
            value = item.value
            if value is Parameter.empty:
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
        for name, item in self._args.items():
            value = item.value
            if value is Parameter.empty:
                value = item.default

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
