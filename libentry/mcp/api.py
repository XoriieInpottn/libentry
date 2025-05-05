#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "APIInfo",
    "route",
    "get",
    "post",
    "list_api_info",
]

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

API_INFO = "__api_info__"


@dataclass
class APIInfo:
    path: str = field()
    methods: List[str] = field()
    chunk_delimiter: str = field(default="\n\n")
    chunk_prefix: str = field(default=None)
    error_prefix: str = field(default="ERROR: ")
    extra_info: Dict[str, Any] = field(default_factory=dict)


def route(
        path: Optional[str] = None,
        methods: List[str] = ("GET", "POST"),
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
        **kwargs
) -> Callable:
    def _api(fn: Callable):
        _path = path
        if _path is None:
            if not hasattr(fn, "__name__"):
                raise RuntimeError("At least one of \"path\" or \"fn.__name__\" should be given.")
            name = getattr(fn, "__name__")
            _path = "/" + name

        setattr(fn, API_INFO, APIInfo(
            methods=methods,
            path=_path,
            chunk_delimiter=chunk_delimiter,
            chunk_prefix=chunk_prefix,
            extra_info=kwargs
        ))
        return fn

    return _api


get = partial(route, methods=["GET"])
post = partial(route, methods=["POST"])


def list_api_info(obj) -> List[Tuple[Callable, APIInfo]]:
    api_list = []
    for name in dir(obj):
        fn = getattr(obj, name)
        if not callable(fn):
            continue
        if not hasattr(fn, API_INFO):
            continue
        api_info = getattr(fn, API_INFO)
        api_list.append((fn, api_info))
    return api_list
