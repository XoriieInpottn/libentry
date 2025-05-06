#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "APIInfo",
    "api",
    "route",
    "get",
    "post",
    "tool",
    "resource",
    "list_api_info",
]

from functools import partial
from typing import Callable, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict

API_INFO_FIELD = "__api_info__"


class APIInfo(BaseModel):
    path: str
    methods: List[Literal["GET", "POST"]]
    tag: Optional[str] = None

    model_config = ConfigDict(extra="allow")


def api(
        path: Optional[str] = None,
        methods: List[Literal["GET", "POST"]] = ("GET", "POST"),
        tag: Optional[str] = None,
        **kwargs
) -> Callable:
    def decorator(fn: Callable):
        _path = path
        if _path is None:
            if not hasattr(fn, "__name__"):
                raise RuntimeError("At least one of \"path\" or \"fn.__name__\" should be given.")
            name = getattr(fn, "__name__")
            _path = "/" + name

        api_info = APIInfo(
            path=_path,
            methods=methods,
            tag=tag
        )
        api_info.model_extra.update(kwargs)
        setattr(fn, API_INFO_FIELD, api_info)
        return fn

    return decorator


route = api
get = partial(api, methods=["GET"])
post = partial(api, methods=["POST"])
tool = partial(api, tag="tool")
resource = partial(api, tag="resource")


def list_api_info(obj) -> List[Tuple[Callable, APIInfo]]:
    api_list = []
    for name in dir(obj):
        fn = getattr(obj, name)
        if not callable(fn):
            continue
        if not hasattr(fn, API_INFO_FIELD):
            continue
        api_info = getattr(fn, API_INFO_FIELD)
        api_list.append((fn, api_info))
    return api_list
