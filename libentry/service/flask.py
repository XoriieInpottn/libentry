#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "run_service",
]

import asyncio
from inspect import signature
from types import GeneratorType
from typing import Callable, Iterable, Type, Union

from flask import Flask, request
from gunicorn.app.base import BaseApplication
from pydantic import BaseModel

from libentry import json
from libentry.api import APIInfo, list_api_info
from libentry.logging import logger


class JSONDumper:

    def __init__(self, api_info: APIInfo):
        self.api_info = api_info

    def dump_stream(self, response: Iterable):
        if self.api_info.stream_prefix is not None:
            yield self.api_info.stream_prefix

        if self.api_info.chunk_delimiter is not None:
            yield self.api_info.chunk_delimiter

        for item in response:
            text = self.dump(item)

            if self.api_info.chunk_prefix is not None:
                yield self.api_info.chunk_prefix

            yield text

            if self.api_info.chunk_suffix is not None:
                yield self.api_info.chunk_suffix

            if self.api_info.chunk_delimiter is not None:
                yield self.api_info.chunk_delimiter

        if self.api_info.stream_suffix is not None:
            yield self.api_info.stream_suffix

        if self.api_info.chunk_delimiter is not None:
            yield self.api_info.chunk_delimiter

    @staticmethod
    def dump(response) -> str:
        if response is None:
            return ""
        elif isinstance(response, BaseModel):
            return json.dumps(response.model_dump())
        else:
            try:
                return json.dumps(response)
            except TypeError:
                return repr(response)


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

        self.dumper = JSONDumper(api_info)

        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__

        self.input_schema = None
        params = signature(fn).parameters
        if len(params) == 1:
            for name, value in params.items():
                if issubclass(value.annotation, BaseModel):
                    self.input_schema = value.annotation

    def __call__(self):
        input_json = json.loads(request.data) if request.data else {}
        if self.input_schema is not None:
            input_data = self.input_schema.model_validate(input_json)
            response = self.fn(input_data)
        else:
            response = self.fn(**input_json)

        if isinstance(response, (GeneratorType, range)):
            return self.app.response_class(
                self.dumper.dump_stream(response),
                mimetype=self.api_info.mime_type
            )
        else:
            return self.app.response_class(
                self.dumper.dump(response),
                mimetype=self.api_info.mime_type
            )


class FlaskServer(Flask):

    def __init__(self, service):
        super().__init__(__name__)
        self.service = service

        logger.info("Initializing Flask application.")
        self.api_info_list = list_api_info(service)
        if len(self.api_info_list) == 0:
            logger.error("No API found, nothing to serve.")
            return

        for fn, api_info in self.api_info_list:
            method = api_info.method
            path = api_info.path
            if asyncio.iscoroutinefunction(fn):
                logger.error(f"Async function \"{fn.__name__}\" is not supported.")
                continue
            logger.info(f"Serving {method}-API for {path}")

            wrapped_fn = FlaskWrapper(self, fn, api_info)
            if method == "GET":
                self.get(path)(wrapped_fn)
            elif method == "POST":
                self.post(path)(wrapped_fn)
            else:
                raise RuntimeError(f"Unsupported method \"{method}\" for ")
        logger.info("Flask application initialized.")

        self.get("/")(self.index)

    def index(self):
        all_api = []
        for _, api_info in self.api_info_list:
            all_api.append({"path": api_info.path})
        return self.response_class(
            json.dumps(all_api, indent=4),
            mimetype="application/json"
        )


class GunicornApplication(BaseApplication):

    def __init__(self, service_type, service_config=None, options=None):
        self.service_type = service_type
        self.service_config = service_config
        self.options = options or {}
        super().__init__()

    def load_config(self):
        config = {
            key: value
            for key, value in self.options.items()
            if key in self.cfg.settings and value is not None
        }
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        logger.info("Initializing the service.")
        if isinstance(self.service_type, type) or callable(self.service_type):
            service = self.service_type(self.service_config) if self.service_config else self.service_type()
        elif self.service_config is None:
            logger.warning(
                "Be careful! It is not recommended to start the server from a service instance. "
                "Use service_type and service_config instead."
            )
            service = self.service_type
        else:
            raise TypeError(f"Invalid service type \"{type(self.service_type)}\".")
        logger.info("Service initialized.")

        return FlaskServer(service)


def run_service(
        service_type: Union[Type, Callable],
        service_config=None,
        host: str = "0.0.0.0",
        port: int = 8888,
        num_workers: int = 1,
        num_threads: int = 10,
        num_connections: int = 2000,
        timeout: int = 60
):
    logger.info("Starting gunicorn server.")
    options = {
        "bind": f"{host}:{port}",
        "workers": num_workers,
        "threads": num_threads,
        "timeout": timeout,
        "worker_connections": num_connections,
    }
    for name, value in options.items():
        logger.info(f"Option {name}: {value}")
    GunicornApplication(service_type, service_config, options).run()
