#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "HasRequestPath",
    "APIInfo",
    "api",
    "route",
    "get",
    "post",
    "tool",
    "resource",
    "list_api_info",
]

import re
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict

API_INFO_FIELD = "__api_info__"

TAG_SUBROUTINE = "subroutine"
TAG_JSONRPC = "jsonrpc"
TAG_ENDPOINT = "endpoint"
TAG_TOOL = "tool"
TAG_RESOURCE = "resource"


class HasRequestPath:
    """The object has a request path.
    A request path is a snake named string starts with "/".
    """

    __request_name__ = None

    @classmethod
    def get_request_path(cls) -> str:
        name = cls.__request_name__
        if name:
            if name.startswith("/"):
                return name
            else:
                return "/" + name
        else:
            name = cls.__name__
            if name.endswith("Request"):
                name = name[:-7]
            return "/" + cls._camel_to_snake(name)

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()


class APIInfo(BaseModel):
    path: str
    methods: List[Literal["GET", "POST"]]
    name: str
    description: Optional[str]
    tag: Optional[str] = None

    model_config = ConfigDict(extra="allow")


def api(
        path: Optional[Union[str, Type[HasRequestPath], HasRequestPath, Any]] = None,
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

        if path:
            if hasattr(path, "get_request_path"):
                _path = path.get_request_path()
            else:
                _path = path
        else:
            _path = f"/{fn_name}"

        if not isinstance(_path, str):
            raise TypeError(f"\"path\" should be instance of str or HasRequestPath.")

        api_info = APIInfo(
            path=_path,
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
