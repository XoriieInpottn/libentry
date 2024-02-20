#!/usr/bin/env python3

"""
@author: xi
@since: 2024-01-08
"""

import re
from dataclasses import is_dataclass, fields
from enum import Enum
from importlib import import_module
from typing import *

__all__ = [
    'DATACLASS_KEY',
    'BASIC_TYPES',
    'get_class_name',
    'import_class_by_name',
    'dataclass_to_dict',
    'dict_to_dataclass',
]

DATACLASS_KEY = '__DATACLASS__'
BASIC_TYPES = (int, float, str, bool)


def get_class_name(obj):
    if not isinstance(obj, type):
        obj = type(obj)
    module_name = obj.__module__
    class_name = obj.__qualname__
    return f'{module_name}.{class_name}'


def import_class_by_name(full_name):
    matched = re.match(r'^(?P<module>[\w.]+)\.(?P<class>\w+)$', full_name)
    assert matched, f'Invalid type name "{full_name}".'
    return getattr(import_module(matched['module']), matched['class'])


def dataclass_to_dict(obj) -> MutableMapping[str, Any]:
    if not is_dataclass(obj):
        raise TypeError('dataclass_to_dict() should be called on dataclass instances.')
    return _dataclass_to_dict(obj)


def _dataclass_to_dict(obj):
    if is_dataclass(obj):
        # obj is a dataclass.
        doc = {DATACLASS_KEY: get_class_name(obj)}
        for f in fields(obj):
            doc[f.name] = _dataclass_to_dict(getattr(obj, f.name))
        return doc
    elif isinstance(obj, tuple) and hasattr(obj, '_fields'):
        # obj is a namedtuple.

        # Recurse into it, but the returned object is another namedtuple of the same type.
        # This is similar to how other list- or tuple-derived classes are treated (see below), but we just need to
        # create them differently because a namedtuple's __init__ needs to be called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict() method, because:
        # - it does not recurse in to the namedtuple fields and convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main use case here is json.dumps, and it handles
        #   converting namedtuples to lists.  Admittedly we're losing some information here when we produce a json list
        #   instead of a dict.  Note that if we returned dicts here instead of namedtuples, we could no longer call
        #   asdict() on a data structure where a namedtuple was used as a dict key.

        return type(obj)(*[_dataclass_to_dict(v) for v in obj])
    elif isinstance(obj, (list, tuple)):
        # obj is a list or tuple.
        return type(obj)(_dataclass_to_dict(v) for v in obj)
    elif isinstance(obj, dict):
        # obj is a dict.
        if not all(isinstance(k, str) for k in obj):
            raise TypeError('All keys of the `dict` should be instance of `str`.')
        return type(obj)((k, _dataclass_to_dict(v)) for k, v in obj.items())
    elif obj is None:
        # obj is None.
        return None
    elif isinstance(obj, BASIC_TYPES):
        # obj is one of the basic types.
        return obj
    elif isinstance(obj, Enum):
        return obj.value
    else:
        raise TypeError(f'`{type(obj)}` is not supported.')


def dict_to_dataclass(doc: Mapping[str, Any]):
    if not (isinstance(doc, dict) and DATACLASS_KEY in doc):
        raise TypeError(f'dict_to_dataclass() should be called on dict with {DATACLASS_KEY} field.')
    return _dict_to_dataclass(doc)


def _dict_to_dataclass(obj):
    if isinstance(obj, dict):
        if DATACLASS_KEY in obj:
            # obj is a dict which is converted from a dataclass.
            dataclass_type = obj[DATACLASS_KEY]
            if isinstance(dataclass_type, str):
                dataclass_type = import_class_by_name(dataclass_type)
            if not is_dataclass(dataclass_type):
                raise TypeError(f'Invalid {DATACLASS_KEY}. Expect str or dataclass, got {type(dataclass_type)}.')
            keys_to_construct = obj.keys() & {_f.name for _f in fields(dataclass_type)}
            kwargs = {k: _dict_to_dataclass(obj[k]) for k in keys_to_construct}
            return dataclass_type(**kwargs)
        else:
            # obj is an ordinary dict.
            return {k: _dict_to_dataclass(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # obj is a list or tuple.
        return type(obj)(_dict_to_dataclass(v) for v in obj)
    elif obj is None:
        # obj is None.
        return None
    elif isinstance(obj, BASIC_TYPES):
        # obj is one of the basic types.
        return obj
    else:
        raise TypeError(f'`{type(obj)}` is not supported.')
