#!/usr/bin/env python3

"""
@author: xi
"""

from dataclasses import dataclass, field
from inspect import signature, Parameter
from typing import *

__all__ = [
    "ConfigurableExecutor",
    "ConfigurableCaller",
    "RegisterDecorator",
    "PolymorphExecutor",
    "Factory",
]


class ConfigurableExecutor(MutableMapping):
    """"""

    @dataclass
    class ArgumentItem:
        value: Any = field(default=Parameter.empty)
        type: Type = field(default=Parameter.empty)
        default: Any = field(default=Parameter.empty)

    def __init__(self, fn: Callable, config=None, **kwargs):
        self._fn = fn

        self._args = {}
        sig = signature(fn.__init__ if isinstance(fn, type) else fn)
        for name, param in sig.parameters.items():
            if isinstance(fn, type) and name == 'self':
                continue
            if param.kind != Parameter.POSITIONAL_OR_KEYWORD:
                continue
            value = param.default
            ann = param.annotation
            if (ann is Parameter.empty) and (value is not Parameter.empty):
                ann = type(param.default)
            self._args[name] = ConfigurableExecutor.ArgumentItem(type=ann, default=value)

        if config is not None:
            self.update_args(config)
        if kwargs:
            self.update_args(kwargs)

    def get_value(self, name: str) -> Any:
        return self._args[name].value

    def get_type(self, name: str) -> Type:
        return self._args[name].type

    def get_default(self, name: str) -> Any:
        return self._args[name].default

    def update_args(self, obj):
        if isinstance(obj, Mapping):
            for name in self:
                if name in obj:
                    setattr(self, name, obj[name])
        else:
            for name in self:
                if hasattr(obj, name):
                    setattr(self, name, getattr(obj, name))

    def execute(self):
        kwargs = {}
        for name, value in self.items():
            if value is Parameter.empty:
                raise ValueError(f'"{name}" must be given.')
            kwargs[name] = value
        return self._fn(**kwargs)

    call = execute
    __call__ = execute

    def __getattr__(self, name):
        item = self._args[name]
        return item.default if item.value is Parameter.empty else item.value

    def __setattr__(self, name, value):
        if name.startswith('_'):
            super(ConfigurableExecutor, self).__setattr__(name, value)
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


ConfigurableCaller = ConfigurableExecutor


class RegisterDecorator(dict):

    def register(self, key, value=None):
        if value is None:
            def _register(_value):
                self[key] = _value
                return _value

            return _register
        else:
            self[key] = value
            return value

    __call__ = register


class PolymorphExecutor(dict):

    def __init__(self, name: str = None):
        super().__init__()
        self.name = name

    def register(self, key: Hashable, fn: Callable = None) -> Callable:
        if fn is None:
            def _register(_fn):
                self[key] = _fn
                return _fn

            return _register
        else:
            self[key] = fn
            return fn

    def execute(self, key: Hashable, __config__=None, **kwargs):
        if key not in self:
            raise RuntimeError(
                (f"{self.name}: \"{key}\"" if self.name else f"\"{key}\"") +
                f" is not implemented."
            )
        executor = ConfigurableExecutor(self[key])
        if __config__ is not None:
            if isinstance(__config__, (list, tuple)):
                for obj in __config__:
                    executor.update_args(obj)
            else:
                executor.update_args(__config__)
        executor.update_args(kwargs)
        return executor.execute()

    call = execute
    __call__ = execute
    build = execute


Factory = PolymorphExecutor
