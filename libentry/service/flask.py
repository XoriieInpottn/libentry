#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "run_service",
]

import asyncio
import re
import traceback
from inspect import signature
from types import GeneratorType
from typing import Any, Callable, Iterable, Optional, Type, Union

from flask import Flask, request
from gunicorn.app.base import BaseApplication
from pydantic import BaseModel, Field, create_model

from libentry import api, json
from libentry.api import APIInfo, list_api_info
from libentry.logging import logger
from libentry.schema import query_api


class JSONDumper:

    def __init__(self, api_info: APIInfo):
        self.api_info = api_info

    def dump_stream(self, response: Iterable) -> Iterable[str]:
        return_value = None

        if self.api_info.stream_prefix is not None:
            yield self.api_info.stream_prefix

            if self.api_info.chunk_delimiter is not None:
                yield self.api_info.chunk_delimiter

        try:
            it = iter(response)
            while True:
                try:
                    item = next(it)
                except StopIteration as e:
                    return_value = e.value
                    break

                text = self.dump(item)

                if self.api_info.chunk_prefix is not None:
                    yield self.api_info.chunk_prefix

                yield text

                if self.api_info.chunk_suffix is not None:
                    yield self.api_info.chunk_suffix

                if self.api_info.chunk_delimiter is not None:
                    yield self.api_info.chunk_delimiter
        except Exception as e:
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise e

            if self.api_info.error_prefix is not None:
                yield self.api_info.error_prefix

            yield self.dump_error(e)

            if self.api_info.chunk_delimiter is not None:
                yield self.api_info.chunk_delimiter

        if self.api_info.stream_suffix is not None:
            yield self.api_info.stream_suffix

            if self.api_info.chunk_delimiter is not None:
                yield self.api_info.chunk_delimiter

        if return_value is not None:
            yield self.dump(return_value)

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

    @staticmethod
    def dump_error(e: Exception) -> str:
        err_cls = e.__class__
        err_name = err_cls.__name__
        module = err_cls.__module__
        if module != "builtins":
            err_name = f"{module}.{err_name}"
        return json.dumps({
            "error": err_name,
            "message": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


def create_model_from_signature(fn):
    sig = signature(fn)

    fields = {}
    for name, param in sig.parameters.items():
        if name in ["self", "cls"]:
            continue
        if param.default is not param.empty:
            fields[name] = (param.annotation, param.default)
        elif param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            fields[name] = (param.annotation, None)
        else:
            fields[name] = (param.annotation, Field())

    return_annotation = sig.return_annotation
    if return_annotation is sig.empty:
        return_annotation = Any
    fields["return"] = (return_annotation, None)
    return create_model(f"__{fn.__name__}_signature", **fields)


class FlaskWrapper:

    def __init__(
            self,
            app: "FlaskServer",
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
        if request.method == "POST":
            input_json = json.loads(request.data) if request.data else {}
        elif request.method == "GET":
            input_json = {**request.args}
        else:
            return self.app.error(f"Unsupported method \"{request.method}\".")

        if self.input_schema is not None:
            try:
                input_data = self.input_schema.model_validate(input_json)
                response = self.fn(input_data)
            except Exception as e:
                if isinstance(e, (SystemExit, KeyboardInterrupt)):
                    raise e
                return self.app.error(self.dumper.dump_error(e))
        else:
            try:
                response = self.fn(**input_json)
            except Exception as e:
                if isinstance(e, (SystemExit, KeyboardInterrupt)):
                    raise e
                return self.app.error(self.dumper.dump_error(e))

        stream = None
        accept = request.headers.get("Accept", "")
        for param in accept.split(";"):
            match = re.search(r"^\s*stream=(.+)$", param)
            if match:
                stream = match.group(1)
                stream = stream in {"1", "true", "True"}
                break

        if stream is not None:
            if stream:
                if not isinstance(response, (GeneratorType, range)):
                    response = [response]
                return self.app.ok(
                    self.dumper.dump_stream(response),
                    mimetype=self.api_info.mime_type
                )
            else:
                if isinstance(response, (GeneratorType, range)):
                    output = []
                    it = iter(response)
                    while True:
                        try:
                            output.append(next(it))
                        except StopIteration as e:
                            if e.value is not None:
                                output.append(e.value)
                            break
                    response = output
                return self.app.ok(
                    self.dumper.dump(response),
                    mimetype=self.api_info.mime_type
                )
        else:
            if isinstance(response, (GeneratorType, range)):
                return self.app.ok(
                    self.dumper.dump_stream(response),
                    mimetype=self.api_info.mime_type
                )
            else:
                return self.app.ok(
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

        for fn, api_info in list_api_info(self):
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

    @api.get("/")
    def index(self, name: str = None):
        if name is None:
            all_api = []
            for _, api_info in self.api_info_list:
                all_api.append({"path": api_info.path})
            return all_api

        for fn, api_info in self.api_info_list:
            if api_info.path == "/" + name:
                return query_api(fn).model_dump()

        return f"No API named \"{name}\""

    @api.get()
    def live(self):
        return "OK"

    def ok(self, body: Union[str, Iterable[str]], mimetype="application/json"):
        return self.response_class(body, status=200, mimetype=mimetype)

    def error(self, body: str, mimetype="text"):
        return self.response_class(body, status=500, mimetype=mimetype)


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
        num_threads: int = 20,
        num_connections: Optional[int] = 100,
        backlog: Optional[int] = 100,
        worker_class: str = "gthread",
        timeout: int = 60,
        keyfile: Optional[str] = None,
        certfile: Optional[str] = None
):
    logger.info("Starting gunicorn server.")
    options = {
        "bind": f"{host}:{port}",
        "workers": num_workers,
        "threads": num_threads,
        "timeout": timeout,
        "worker_connections": num_connections if num_connections else num_threads,
        "backlog": backlog if backlog else num_threads,
        "keyfile": keyfile,
        "certfile": certfile,
        "worker_class": worker_class
    }
    for name, value in options.items():
        logger.info(f"Option {name}: {value}")
    GunicornApplication(service_type, service_config, options).run()
