#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "dump",
    "dumps",
    "load",
    "loads",
]

import json
from base64 import b64decode, b64encode
from functools import partial

import numpy as np
from pydantic import BaseModel

_BINDINGS = []


def bind(name, type_):
    def _wrapper(cls):
        _BINDINGS.append((name, type_, cls()))
        return cls

    return _wrapper


def custom_encode(o) -> dict:
    if isinstance(o, BaseModel):
        return o.model_dump()

    for name, type_, support in _BINDINGS:
        if isinstance(o, type_):
            doc = support.encode(o)
            doc["__TYPE__"] = name
            return doc
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def custom_decode(d: dict):
    __type__ = d.get("__TYPE__")
    if __type__ is None:
        return d

    for name, _, support in _BINDINGS:
        if __type__ == name:
            return support.decode(d)
    return d


dump = partial(json.dump, default=custom_encode, ensure_ascii=False)
dumps = partial(json.dumps, default=custom_encode, ensure_ascii=False)

load = partial(json.load, object_hook=custom_decode)
loads = partial(json.loads, object_hook=custom_decode)


@bind("bytes", bytes)
class BytesSupport:

    def encode(self, o: bytes) -> dict:
        return {
            "data": b64encode(o).decode(),
        }

    def decode(self, d: dict) -> bytes:
        return b64decode(d["data"])


@bind("numpy", np.ndarray)
class NumpySupport:

    def encode(self, a):
        shape = a.shape
        dtype = a.dtype
        data = a.tobytes(order="C")
        data = b64encode(data).decode()
        return {
            "data": data,
            "shape": shape,
            "dtype": str(dtype),
        }

    def decode(self, d):
        shape = d["shape"]
        dtype = d["dtype"]
        buffer = b64decode(d["data"].encode())
        return np.frombuffer(buffer, dtype=dtype).reshape(shape)
