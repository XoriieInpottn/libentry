#!/usr/bin/env python3

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


def encode_ndarray(a: np.ndarray):
    shape = a.shape
    dtype = a.dtype
    data = a.tobytes(order="C")
    data = b64encode(data).decode()
    return {
        "__TYPE__": "numpy",
        "shape": shape,
        "dtype": str(dtype),
        "data": data
    }


def decode_ndarray(d):
    shape = d["shape"]
    dtype = d["dtype"]
    buffer = b64decode(d["data"].encode())
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)


def custom_encode(o):
    if isinstance(o, np.ndarray):
        return encode_ndarray(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def custom_decode(d):
    __type__ = d.get("__TYPE__")
    if __type__ is None:
        return d
    if __type__ == "numpy":
        return decode_ndarray(d)
    else:
        return d


dump = partial(json.dump, default=custom_encode)
dumps = partial(json.dumps, default=custom_encode)

load = partial(json.load, object_hook=custom_decode)
loads = partial(json.loads, object_hook=custom_decode)
