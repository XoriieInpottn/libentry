#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "Client",
]

import os
from typing import Mapping, Optional, Union

import json5
import requests
from pydantic import BaseModel


class Client:

    def __init__(
            self,
            base_url: str,
            api_key: str = None,
            user_agent: str = "API Client",
            verify=True
    ) -> None:
        self.base_url = base_url
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        }
        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.verify = verify

    def get(self, path: str):
        api_url = os.path.join(self.base_url, path)
        resp = requests.get(api_url, headers=self.headers, verify=self.verify)
        text = resp.text
        try:
            return json5.loads(text)
        except ValueError:
            return text

    def post(self, path: str, json: Mapping = None, stream=False, stream_delimiter: str = "\n\n"):
        full_url = os.path.join(self.base_url, path)
        resp = requests.post(
            full_url,
            headers=self.headers,
            json=json if json is not None else {},
            verify=self.verify,
            stream=stream
        )
        if resp.status_code != 200:
            raise RuntimeError(resp.text)

        if stream:
            if stream_delimiter is None:
                return resp.iter_content()
            else:
                return resp.iter_lines(delimiter=stream_delimiter.encode())
        else:
            text = resp.text
            try:
                return json5.loads(text)
            except ValueError:
                return text

    def __getattr__(self, item: str):
        return MethodProxy(self, item)


class MethodProxy:

    def __init__(self, client: Client, url: str):
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
