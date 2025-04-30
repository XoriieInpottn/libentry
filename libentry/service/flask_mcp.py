#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "MCPAdapter",
    "JSONAdapter",
    "FlaskFunction",
    "SSEMixIn",
    "LifeCycleMixIn",
    "ToolsMixIn",
    "FlaskServer",
    "GunicornApplication",
    "run_service",
]

import asyncio
import traceback
import uuid
from inspect import signature
from queue import Empty, Queue
from threading import Lock
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, Optional, Type, Union

from flask import Flask, request as flask_request
from pydantic import BaseModel, TypeAdapter

from libentry import api_mcp as api, json, logger
from libentry.api_mcp import APIInfo, ContentType, JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse, \
    SSEEvent, list_api_info
from libentry.schema import query_api

try:
    from gunicorn.app.base import BaseApplication
except ImportError:
    class BaseApplication:

        def load(self) -> Flask:
            pass

        def run(self):
            flask_server = self.load()
            assert hasattr(self, "options")
            bind = getattr(self, "options")["bind"]
            pos = bind.rfind(":")
            host = bind[:pos]
            port = int(bind[pos + 1:])
            logger.warn("Your system doesn't support gunicorn.")
            logger.warn("Use Flask directly.")
            logger.warn("Options like \"num_threads\", \"num_workers\" are ignored.")
            return flask_server.run(host=host, port=port)


class MCPAdapter:

    def __init__(self, fn: Callable):
        self.fn = fn
        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__

        self.input_schema = None
        params = signature(fn).parameters
        if len(params) == 1:
            for name, value in params.items():
                annotation = value.annotation
                if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                    self.input_schema = annotation

    def __call__(
            self,
            request: Union[JSONRPCNotification, JSONRPCRequest]
    ) -> Union[JSONRPCResponse, Iterable[SSEEvent], None]:
        params = request.params if request.params is not None else {}

        try:
            result = self._call_fn(params)
        except SystemExit as e:
            raise e
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if isinstance(request, JSONRPCRequest):
                return JSONRPCResponse(
                    jsonrpc=request.jsonrpc,
                    id=request.id,
                    error=JSONRPCError.from_exception(e)
                )
            else:
                return None

        if isinstance(request, JSONRPCRequest):
            if not isinstance(result, (GeneratorType, range)):
                return JSONRPCResponse(
                    jsonrpc=request.jsonrpc,
                    id=request.id,
                    result=result
                )
            else:
                return (
                    SSEEvent(
                        event="message",
                        data=JSONRPCResponse(
                            jsonrpc=request.jsonrpc,
                            id=request.id,
                            result=item
                        )
                    )
                    for item in result
                )
        else:
            return None

    def _call_fn(self, params: Dict[str, Any]) -> Any:
        if self.input_schema is not None:
            # Note that "input_schema is not None" means:
            # (1) The function has only one argument;
            # (2) The arguments is a BaseModel.
            # In this case, the request data can be directly validated as a "BaseModel" and
            # subsequently passed to the function as a single object.
            pydantic_params = self.input_schema.model_validate(params)
            return self.fn(pydantic_params)
        else:
            # The function has multiple arguments, and the request data bundle them as a single object.
            # So, they should be unpacked before pass to the function.
            return self.fn(**params)

    @staticmethod
    def _make_error(e):
        err_cls = e.__class__
        err_name = err_cls.__name__
        module = err_cls.__module__
        if module != "builtins":
            err_name = f"{module}.{err_name}"
        return {
            "code": 1,
            "message": f"{err_name}: {str(e)}",
            "data": traceback.format_exc()
        }


class JSONAdapter:

    def __init__(self, fn: Callable):
        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__
        self.fn = MCPAdapter(fn) if not isinstance(fn, MCPAdapter) else fn
        self.type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCNotification])

    def __call__(self, request: Union[Dict[str, Any], JSONRPCRequest, JSONRPCNotification]):
        if not isinstance(request, (JSONRPCRequest, JSONRPCNotification)):
            request = self.type_adapter.validate_python(request)
        return self.fn(request)


class FlaskFunction:

    def __init__(self, fn: Callable, api_info: APIInfo, app: "FlaskServer"):
        if not api_info.extra_info.get("skip_adapter", False):
            fn = JSONAdapter(fn)
        self.fn = fn
        self.api_info = api_info
        self.app = app
        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__

    def __call__(self):
        args = flask_request.args
        data = flask_request.data
        content_type = flask_request.content_type

        json_from_url = {**args}
        if data:
            if (not content_type) or content_type == ContentType.json.value:
                json_from_data = json.loads(data)
            else:
                return self.app.error(f"Unsupported Content-Type: \"{content_type}\".")
        else:
            json_from_data = {}

        conflicts = json_from_url.keys() & json_from_data.keys()
        if len(conflicts) > 0:
            return self.app.error(f"Duplicated fields: \"{conflicts}\".")

        input_json = {**json_from_url, **json_from_data}

        ################################################################################
        # Call method as MCP
        ################################################################################
        try:
            mcp_response = self.fn(input_json)
        except Exception as e:
            return self.app.error(str(e))

        ################################################################################
        # Parse MCP response
        ################################################################################
        accepts = flask_request.accept_mimetypes
        print([*accepts])
        if mcp_response is None:
            return self.app.ok(
                None,
                mimetype=ContentType.json.value
            )
        elif isinstance(mcp_response, JSONRPCResponse):
            return self.app.ok(
                json.dumps(mcp_response.model_dump(exclude_none=True)),
                mimetype=ContentType.json.value
            )
        elif isinstance(mcp_response, Dict):
            return self.app.ok(
                json.dumps(mcp_response),
                mimetype=ContentType.json.value
            )
        elif isinstance(mcp_response, (GeneratorType, range)):
            if ContentType.sse.value in accepts:
                return self.app.ok(
                    self._sse_stream(mcp_response),
                    mimetype=ContentType.sse.value
                )
            else:
                return self.app.error(f"Unsupported Accept: \"{[*accepts]}\".")
        else:
            return self.app.ok(
                str(mcp_response),
                mimetype=ContentType.json.value
            )

    def _sse_stream(self, events: Iterable):
        for item in events:
            if not isinstance(item, SSEEvent):
                item = SSEEvent.model_validate(item)
            yield "event:"
            yield item.event
            d = item.data
            if d is not None:
                yield "\n"
                yield "data:"
                if isinstance(d, BaseModel):
                    yield json.dumps(d.model_dump(exclude_none=True))
                elif isinstance(d, Dict):
                    yield json.dumps(d)
                else:
                    yield str(d)
            yield self.api_info.chunk_delimiter


class SSEMixIn:

    def __init__(self):
        self.lock = Lock()
        self.sse_dict = {}

    @api.get("/sse", skip_adapter=True)
    def sse(self, _) -> Iterable[SSEEvent]:
        session_id = str(uuid.uuid4())
        queue = Queue(8)
        with self.lock:
            self.sse_dict[session_id] = queue

        def _stream():
            yield SSEEvent(event="endpoint", data=f"/message?sessionId={session_id}")
            try:
                while True:
                    try:
                        message = queue.get(timeout=3)
                        if message is None:
                            break
                        yield SSEEvent(event="message", data=message)
                    except Empty:
                        yield SSEEvent(event="message", data=JSONRPCRequest(id=0, method="ping"))
            finally:
                with self.lock:
                    del self.sse_dict[session_id]
                logger.info(f"Session {session_id} cleaned.")

        return _stream()

    @api.route("/message", skip_adapter=True)
    def message(self, request: Dict[str, Any]) -> None:
        ################################################################################
        # session validation
        ################################################################################
        session_id = request.get("sessionId")
        if session_id is None:
            raise RuntimeError("You should start a session by request the \"/sse\" endpoint first.")
        with self.lock:
            if session_id not in self.sse_dict:
                raise RuntimeError(f"Invalid session: \"{session_id}\".")

        ################################################################################
        # validate request
        ################################################################################
        type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCNotification])
        mcp_request = type_adapter.validate_python(request)
        print("/message", mcp_request)

        ################################################################################
        # call the mcp method
        ################################################################################
        path = "/" + mcp_request.method
        service_routes = getattr(self, "service_routes")
        internal_routes = getattr(self, "internal_routes")
        route = service_routes.get(path, internal_routes.get(path))
        if route is None:
            raise RuntimeError(f"Method {mcp_request.method} doesn't exist.")

        response = route["flask_fn"].fn(mcp_request)

        ################################################################################
        # put response
        ################################################################################
        with self.lock:
            queue = self.sse_dict[session_id]

        if isinstance(response, (GeneratorType, range)):
            for item in response:
                queue.put(item)
        elif response is not None:
            queue.put(response)


class LifeCycleMixIn:

    @api.route()
    def live(self):
        return "OK"

    @api.route()
    def initialize(self, protocolVersion: str, capabilities, clientInfo):
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"logging": {}, "tools": {}},
            "serverInfo": {"name": "libentry-server", "version": "1.0.0"}
        }

    @api.route("/notifications/initialized")
    def notifications_initialized(self):
        pass


class ToolsMixIn:

    @api.route("/tools/list")
    def tools_list(self):
        service_routes: dict = getattr(self, "service_routes")

        tools = []
        for route in service_routes.values():
            api_info = route["api_info"]
            fn = route["fn"]
            tool_item = {
                "name": api_info.path,
                "description": "NO",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            tools.append(tool_item)
            schema = query_api(fn)
            input_schema = schema.context[schema.input_schema]
            for field in input_schema.fields:
                tool_item["inputSchema"]["properties"][field.name] = {
                    "type": field.type,
                    "description": field.description
                }
                if field.is_required:
                    tool_item["inputSchema"]["required"].append(field.name)
        return {"tools": tools}

    @api.route("/tools/call")
    def tools_call(self, name: str, arguments: dict):
        path = "/" + name
        service_routes: dict = getattr(self, "service_routes")
        route = service_routes[path]
        flask_fn: FlaskFunction = route["flask_fn"]
        mcp_fn = flask_fn.fn
        if isinstance(mcp_fn, JSONAdapter):
            mcp_fn = mcp_fn.fn

        tool_request = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=name,
            params=arguments
        )
        tool_response = mcp_fn(tool_request)

        if tool_response.result is not None:
            result = tool_response.result
            text = json.dumps(result) if isinstance(result, Dict) else str(result)
            response = {
                "content": [{"type": "text", "text": text}],
                "isError": False
            }
        elif tool_response.error is not None:
            result = tool_response.error
            text = json.dumps(result) if isinstance(result, Dict) else str(result)
            response = {
                "content": [{"type": "text", "text": text}],
                "isError": True
            }
        else:
            response = {
                "content": [{"type": "text", "text": "Unknown error."}],
                "isError": True
            }
        return response


class FlaskServer(Flask, SSEMixIn, LifeCycleMixIn, ToolsMixIn):

    def __init__(self, service):
        Flask.__init__(self, __name__)
        SSEMixIn.__init__(self)
        LifeCycleMixIn.__init__(self)
        ToolsMixIn.__init__(self)

        self.service = service

        logger.info("Initializing Flask application.")
        self.service_routes = {
            api_info.path: {"api_info": api_info, "fn": fn}
            for fn, api_info in list_api_info(self.service)
        }
        self._init_service_routes()

        self.internal_routes = {
            api_info.path: {"api_info": api_info, "fn": fn}
            for fn, api_info in list_api_info(self)
        }
        self._init_internal_routes()
        logger.info("Flask application initialized.")

    def _init_service_routes(self):
        for route in self.service_routes.values():
            api_info = route["api_info"]
            fn = route["fn"]
            methods = api_info.methods
            path = api_info.path
            if asyncio.iscoroutinefunction(fn):
                logger.error(f"Async function \"{fn.__name__}\" is not supported.")
                continue
            logger.info(f"Serving {path} as {', '.join(methods)}.")

            flask_fn = FlaskFunction(fn, api_info, self)
            route["flask_fn"] = flask_fn
            self.route(path, methods=methods)(flask_fn)

    def _init_internal_routes(self):
        for route in self.internal_routes.values():
            api_info = route["api_info"]
            fn = route["fn"]
            methods = api_info.methods
            path = api_info.path

            if api_info.path in self.service_routes:
                logger.info(f"Use custom implementation of {path}.")
                continue

            if asyncio.iscoroutinefunction(fn):
                logger.error(f"Async function \"{fn.__name__}\" is not supported.")
                continue
            logger.info(f"Serving {path} as {', '.join(methods)}.")

            flask_fn = FlaskFunction(fn, api_info, self)
            route["flask_fn"] = flask_fn
            self.route(path, methods=methods)(flask_fn)

    def ok(self, body: Union[str, Iterable[str], None], mimetype: str):
        return self.response_class(body, status=200, mimetype=mimetype)

    def error(self, body: str, mimetype=ContentType.plain.value):
        return self.response_class(body, status=500, mimetype=mimetype)

    @api.get("/")
    def index(self, name: str = None):
        if name is None:
            all_api = []
            for route in self.service_routes.values():
                api_info = route["api_info"]
                all_api.append({"path": api_info.path})
            return all_api

        path = "/" + name
        if path in self.service_routes:
            fn = self.service_routes[path]["fn"]
            return query_api(fn).model_dump()
        else:
            return f"No API named \"{name}\""


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
        num_connections: Optional[int] = 1000,
        backlog: Optional[int] = 1000,
        worker_class: str = "gthread",
        timeout: int = 60,
        keyfile: Optional[str] = None,
        keyfile_password: Optional[str] = None,
        certfile: Optional[str] = None
):
    logger.info("Starting gunicorn server.")
    if num_connections is None or num_connections < num_threads * 2:
        num_connections = num_threads * 2
    if backlog is None or backlog < num_threads * 2:
        backlog = num_threads * 2

    def ssl_context(config, _default_ssl_context_factory):
        import ssl
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(
            certfile=config.certfile,
            keyfile=config.keyfile,
            password=keyfile_password
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        return context

    options = {
        "bind": f"{host}:{port}",
        "workers": num_workers,
        "threads": num_threads,
        "timeout": timeout,
        "worker_connections": num_connections,
        "backlog": backlog,
        "keyfile": keyfile,
        "certfile": certfile,
        "worker_class": worker_class,
        "ssl_context": ssl_context
    }
    for name, value in options.items():
        logger.info(f"Option {name}: {value}")
    GunicornApplication(service_type, service_config, options).run()
