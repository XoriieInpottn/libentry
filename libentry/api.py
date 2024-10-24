#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "APIInfo",
    "api",
    "get",
    "post",
    "list_api_info",
    "APIClient",
]

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Literal, Mapping, Optional, Tuple, Union

import requests
from pydantic import BaseModel

from libentry import json

API_INFO = "__api_info__"


@dataclass
class APIInfo:
    method: str = field()
    path: str = field()
    mime_type: str = field(default="application/json")
    chunk_delimiter: str = field(default="\n\n")
    chunk_prefix: str = field(default=None)
    chunk_suffix: str = field(default=None)
    stream_prefix: str = field(default=None)
    stream_suffix: str = field(default=None)
    extra_info: Mapping[str, Any] = field(default_factory=dict)


def api(
        method: Literal["GET", "POST"] = "POST",
        path: Optional[str] = None,
        mime_type: str = "application/json",
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
        chunk_suffix: str = None,
        stream_prefix: str = None,
        stream_suffix: str = None,
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
            method=method,
            path=_path,
            mime_type=mime_type,
            chunk_delimiter=chunk_delimiter,
            chunk_prefix=chunk_prefix,
            chunk_suffix=chunk_suffix,
            stream_prefix=stream_prefix,
            stream_suffix=stream_suffix,
            extra_info=kwargs
        ))
        return fn

    return _api


def get(
        path: Optional[str] = None,
        mime_type: str = "application/json",
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
        chunk_suffix: str = None,
        stream_prefix: str = None,
        stream_suffix: str = None,
        **kwargs
) -> Callable:
    return api(
        method="GET",
        path=path,
        mime_type=mime_type,
        chunk_delimiter=chunk_delimiter,
        chunk_prefix=chunk_prefix,
        chunk_suffix=chunk_suffix,
        stream_prefix=stream_prefix,
        stream_suffix=stream_suffix,
        **kwargs
    )


def post(
        path: Optional[str] = None,
        mime_type: str = "application/json",
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
        chunk_suffix: str = None,
        stream_prefix: str = None,
        stream_suffix: str = None,
        **kwargs
) -> Callable:
    return api(
        method="POST",
        path=path,
        mime_type=mime_type,
        chunk_delimiter=chunk_delimiter,
        chunk_prefix=chunk_prefix,
        chunk_suffix=chunk_suffix,
        stream_prefix=stream_prefix,
        stream_suffix=stream_suffix,
        **kwargs
    )


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


class APIClient:

    def __init__(
            self,
            base_url: str,
            api_key: str = None,
            accept: str = "application/json",
            content_type: str = "application/json",
            user_agent: str = "API Client",
            connection: str = "close",
            verify=False,
            **extra_headers
    ) -> None:
        self.base_url = base_url
        self.headers = {
            "Accept": accept,
            "Content-Type": content_type,
            "User-Agent": user_agent,
            "Connection": connection,
            **extra_headers
        }
        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.verify = verify

    def get(self, path: str, timeout=60):
        api_url = os.path.join(self.base_url, path)
        response = requests.get(api_url, headers=self.headers, verify=self.verify, timeout=timeout)

        if response.status_code != 200:
            raise RuntimeError(response.text)

        try:
            return self._load_json(response.text)
        finally:
            response.close()

    def post(
            self,
            path: str,
            json_data: Mapping = None,
            stream=False,
            timeout=60,
            chunk_delimiter: str = "\n\n",
            chunk_prefix: str = None,
            chunk_suffix: str = None,
    ):
        full_url = os.path.join(self.base_url, path)

        data = json.dumps(json_data) if json_data is not None else None
        response = requests.post(
            full_url,
            headers=self.headers,
            data=data,
            verify=self.verify,
            stream=stream,
            timeout=timeout
        )
        if response.status_code != 200:
            raise RuntimeError(response.text)

        if stream:
            if chunk_delimiter is None:
                # TODO: this branch is not tested yet!
                return response.iter_content(decode_unicode=True)
            else:
                return self._iter_chunks(
                    response=response,
                    chunk_delimiter=chunk_delimiter.encode() if chunk_delimiter else None,
                    chunk_prefix=chunk_prefix.encode() if chunk_prefix else None,
                    chunk_suffix=chunk_suffix.encode() if chunk_suffix else None,
                )
        else:
            try:
                return self._load_json(response.text)
            finally:
                response.close()

    def _iter_chunks(
            self,
            response: requests.Response,
            chunk_delimiter: bytes,
            chunk_prefix: bytes,
            chunk_suffix: bytes
    ) -> Iterable:
        try:
            for chunk in response.iter_lines(decode_unicode=False, delimiter=chunk_delimiter):
                if not chunk:
                    continue

                if chunk_prefix is not None:
                    if chunk.startswith(chunk_prefix):
                        chunk = chunk[len(chunk_prefix):]
                    else:
                        continue

                if chunk_suffix is not None:
                    if chunk.endswith(chunk_suffix):
                        chunk = chunk[:-len(chunk_suffix)]
                    else:
                        continue

                yield self._load_json(chunk)
        finally:
            response.close()

    @staticmethod
    def _load_json(text: str):
        try:
            return json.loads(text)
        except ValueError:
            return text

    def __getattr__(self, item: str):
        return MethodProxy(self, item)


class MethodProxy:

    def __init__(self, client: APIClient, url: str):
        self.client = client
        self.url = url

    def __call__(self, request: Optional[Union[Mapping, BaseModel]] = None, **kwargs):
        if request is None:
            request = kwargs
        elif isinstance(request, BaseModel):
            request = request.model_dump()
        for k, v in kwargs.items():
            if isinstance(v, BaseModel):
                v = v.model_dump()
            request[k] = v
        return self.client.post(self.url, request)
