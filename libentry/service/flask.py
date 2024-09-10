#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "run_service",
]

import json
from inspect import signature
from types import GeneratorType
from typing import Callable, Iterable

from flask import Flask, request
from pydantic import BaseModel

from libentry.api import APIInfo, list_api_info


class FlaskWrapper:

    def __init__(
            self,
            app: Flask,
            fn: Callable,
            api_info: APIInfo
    ):
        self.app = app
        self.fn = fn
        self.api_info = api_info

        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__

        self.input_schema = None
        params = signature(fn).parameters
        if len(params) == 1:
            for name, value in params.items():
                if issubclass(value.annotation, BaseModel):
                    self.input_schema = value.annotation

    def __call__(self):
        input_json = request.json
        if self.input_schema is not None:
            input_data = self.input_schema.model_validate(input_json)
            response = self.fn(input_data)
        else:
            response = self.fn(**input_json)

        if isinstance(response, (GeneratorType, range)):
            stream = self._dump_stream_response(response)
            return self.app.response_class(stream, mimetype="text")
        else:
            return self._dump_response(response)

    def _dump_stream_response(self, response: Iterable):
        for item in response:
            item = self._dump_response(item)
            if self.api_info.stream_prefix is not None:
                item = self.api_info.stream_prefix + item
            if self.api_info.stream_delimiter is not None:
                item = item + self.api_info.stream_delimiter
            yield item

    @staticmethod
    def _dump_response(response) -> str:
        if response is None:
            return ""
        elif isinstance(response, BaseModel):
            return response.model_dump_json()
        else:
            try:
                return json.dumps(response)
            except TypeError:
                return repr(response)


def run_service(service, host: str, port: int):
    api_info_list = list_api_info(service)
    if len(api_info_list) == 0:
        print("No API found, nothing to serve.")
        return

    app = Flask(__name__)
    for fn, api_info in api_info_list:
        method = api_info.method
        path = api_info.path
        print(method, path)

        wrapped_fn = FlaskWrapper(app, fn, api_info)
        if method == "GET":
            app.get(path)(wrapped_fn)
        elif method == "POST":
            app.post(path)(wrapped_fn)
        else:
            raise RuntimeError(f"Unsupported method \"{method}\" for ")
    app.run(host, port, threaded=True)
