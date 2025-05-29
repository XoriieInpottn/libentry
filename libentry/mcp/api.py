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

TAG_SUBROUTINE = "subroutine"
TAG_JSONRPC = "jsonrpc"
TAG_ENDPOINT = "endpoint"
TAG_TOOL = "tool"
TAG_RESOURCE = "resource"


class APIInfo(BaseModel):
    path: str
    methods: List[Literal["GET", "POST"]]
    name: str
    description: Optional[str]
    tag: Optional[str] = None

    model_config = ConfigDict(extra="allow")


def api(
        path: Optional[str] = None,
        methods: List[Literal["GET", "POST"]] = ("GET", "POST"),
        name: Optional[str] = None,
        description: Optional[str] = None,
        tag: Optional[str] = None,
        **kwargs
) -> Callable:
    def decorator(fn: Callable):
        if not hasattr(fn, "__name__"):
            raise RuntimeError("At least one of \"path\" or \"fn.__name__\" should be given.")
        fn_name = getattr(fn, "__name__")
        fn_doc = getattr(fn, "__doc__")

        api_info = APIInfo(
            path=path or f"/{fn_name}",
            methods=methods,
            name=name or fn_name,
            description=description or fn_doc,
            tag=tag
        )
        api_info.model_extra.update(kwargs)
        setattr(fn, API_INFO_FIELD, api_info)
        return fn

    return decorator


route = api
get = partial(api, methods=["GET"])
post = partial(api, methods=["POST"])


def tool(path=None, name=None, description=None, **kwargs):
    return api(
        path=path,
        methods=["POST"],
        name=name,
        description=description,
        tag=TAG_TOOL,
        **kwargs
    )


def resource(uri, name=None, description=None, mime_type=None, size=None, **kwargs):
    return api(
        path=None,
        methods=["POST"],
        name=name,
        description=description,
        tag=TAG_RESOURCE,
        mimeType=mime_type,
        size=size,
        uri=uri,
        **kwargs
    )


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
