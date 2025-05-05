#!/usr/bin/env python3

__author__ = "xi"

import asyncio
import uuid
from dataclasses import dataclass
from inspect import signature
from queue import Empty, Queue
from threading import Lock
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, Optional, Type, Union

from flask import Flask, request as flask_request
from pydantic import BaseModel, TypeAdapter

from libentry import json, logger
from libentry.mcp import api
from libentry.mcp.api import APIInfo, list_api_info
from libentry.mcp.types import CallToolRequestParams, CallToolResult, ContentType, Implementation, \
    InitializeRequestParams, \
    InitializeResult, JSONRPCError, \
    JSONRPCNotification, JSONRPCRequest, \
    JSONRPCResponse, ListToolsResult, SSE, ServerCapabilities, TextContent, Tool, ToolProperty, ToolSchema, \
    ToolsCapability
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


class JSONRPCAdapter:

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
        self.type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCNotification])

    def __call__(
            self,
            request: Union[JSONRPCRequest, JSONRPCNotification, Dict[str, Any]]
    ) -> Union[JSONRPCResponse, Iterable[JSONRPCResponse], None]:
        if isinstance(request, Dict):
            request = self.type_adapter.validate_python(request)

        try:
            result = self._call_fn(request)
        except SystemExit as e:
            raise e
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if isinstance(request, JSONRPCNotification):
                return None

            return JSONRPCResponse(
                jsonrpc=request.jsonrpc,
                id=request.id,
                error=JSONRPCError.from_exception(e)
            )

        if isinstance(request, JSONRPCNotification):
            return None

        if not isinstance(result, (GeneratorType, range)):
            return (
                result if isinstance(result, JSONRPCResponse) else
                JSONRPCResponse(id=request.id, result=result)
            )
        else:
            return (
                (item if isinstance(item, JSONRPCResponse) else
                 JSONRPCResponse(id=request.id, result=item))
                for item in result
            )

    def _call_fn(self, request: Union[JSONRPCRequest, JSONRPCNotification]) -> Any:
        if issubclass(self.input_schema, (JSONRPCResponse, JSONRPCNotification)):
            raw_params = self.input_schema.model_validate(request.model_dump())
            return self.fn(raw_params)
        elif issubclass(self.input_schema, BaseModel):
            # Note that "input_schema is not None" means:
            # (1) The function has only one argument;
            # (2) The arguments is a BaseModel.
            # In this case, the request data can be directly validated as a "BaseModel" and
            # subsequently passed to the function as a single object.
            pydantic_params = self.input_schema.model_validate(request.params or {})
            return self.fn(pydantic_params)
        else:
            # The function has multiple arguments, and the request data bundle them as a single object.
            # So, they should be unpacked before pass to the function.
            kwargs = request.params or {}
            return self.fn(**kwargs)


class FlaskHandler:

    def __init__(self, fn: Callable, api_info: APIInfo, app: "FlaskServer"):
        if not api_info.extra_info.get("skip_adapter", False):
            fn = JSONRPCAdapter(fn)
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
        print(input_json, [*accepts])
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
            if isinstance(item, SSE):
                event = item.event
                data = item.data
            else:
                event = "message"
                data = item
            yield "event:"
            yield event
            if data is not None:
                yield "\n"
                yield "data:"
                if isinstance(data, BaseModel):
                    yield json.dumps(data.model_dump(exclude_none=True))
                elif isinstance(data, Dict):
                    yield json.dumps(data)
                else:
                    yield str(data)
            yield "\n\n"


class SSEMixIn:

    def __init__(self):
        self.lock = Lock()
        self.sse_dict = {}

    @api.get("/sse", skip_adapter=True)
    def sse(self, _) -> Iterable[SSE]:
        session_id = str(uuid.uuid4())
        queue = Queue(8)
        with self.lock:
            self.sse_dict[session_id] = queue

        def _stream():
            yield SSE(event="endpoint", data=f"/message?sessionId={session_id}")
            try:
                while True:
                    try:
                        message = queue.get(timeout=3)
                        if message is None:
                            break
                        yield SSE(event="message", data=message)
                    except Empty:
                        yield SSE(event="message", data=JSONRPCRequest(id=0, method="ping"))
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
        service_routes: Dict[str, Route] = getattr(self, "service_routes")
        internal_routes: Dict[str, Route] = getattr(self, "internal_routes")
        route = service_routes.get(path, internal_routes.get(path))
        if route is None:
            raise RuntimeError(f"Method {mcp_request.method} doesn't exist.")

        response = route.handler.fn(mcp_request)

        ################################################################################
        # put response
        ################################################################################
        with self.lock:
            queue = self.sse_dict[session_id]

        print(response)
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
    def initialize(self, _params: InitializeRequestParams) -> InitializeResult:
        return InitializeResult(
            protocolVersion="2024-11-05",
            capabilities=ServerCapabilities(tools=ToolsCapability(listChanged=False)),
            serverInfo=Implementation(name="python-libentry", version="1.0.0")
        )


class NotificationsMixIn:

    @api.route("/notifications/initialized")
    def initialized(self):
        pass


class ToolsMixIn:

    def __init__(self):
        self.service_routes: dict[str, Route] = {}

    @api.route("/tools/list")
    def list(self) -> ListToolsResult:
        tools = []
        for route in self.service_routes.values():
            api_info = route.api_info
            fn = route.fn
            tool = Tool(
                name=api_info.path[1:],
                description=None,
                inputSchema=ToolSchema()
            )
            tools.append(tool)
            schema = query_api(fn)
            input_schema = schema.context[schema.input_schema]
            for field in input_schema.fields:
                tool.inputSchema.properties[field.name] = ToolProperty(
                    type=field.type,
                    description=field.description
                )
                if field.is_required:
                    tool.inputSchema.required.append(field.name)
        return ListToolsResult(tools=tools)

    @api.route("/tools/call")
    def call(self, params: CallToolRequestParams) -> CallToolResult:
        route = self.service_routes.get(f"/{params.name}")
        if route is None:
            return CallToolResult(
                content=[TextContent(text=f"Tool \"{params.name}\" doesn't exist.")],
                isError=True
            )

        rpc_fn = route.handler.fn
        rpc_request = JSONRPCRequest(
            id=str(uuid.uuid4()),
            method=params.name,
            params=params.arguments
        )
        rpc_response = rpc_fn(rpc_request)

        if rpc_response.result is not None:
            result = rpc_response.result
            text = json.dumps(result) if isinstance(result, Dict) else str(result)
            return CallToolResult(
                content=[TextContent(text=text)],
                isError=False
            )
        elif rpc_response.error is not None:
            error = rpc_response.error
            text = json.dumps(error) if isinstance(error, Dict) else str(error)
            return CallToolResult(
                content=[TextContent(text=text)],
                isError=True
            )
        else:
            return CallToolResult(
                content=[TextContent(text="Unknown error.")],
                isError=True
            )


@dataclass
class Route:
    api_info: APIInfo
    fn: Callable
    handler: FlaskHandler


class FlaskServer(Flask, SSEMixIn, LifeCycleMixIn, NotificationsMixIn, ToolsMixIn):

    def __init__(self, service):
        Flask.__init__(self, __name__)
        SSEMixIn.__init__(self)
        LifeCycleMixIn.__init__(self)
        NotificationsMixIn.__init__(self)
        ToolsMixIn.__init__(self)

        self.service = service

        logger.info("Initializing Flask application.")
        self.service_routes = self._create_routes(self.service)
        self.internal_routes = self._create_routes(self, self.service_routes)
        for route in (*self.service_routes.values(), *self.internal_routes.values()):
            self.route(route.api_info.path, methods=route.api_info.methods)(route.handler)
        logger.info("Flask application initialized.")

    def _create_routes(self, service: Any, existing: Optional[Dict[str, Route]] = None) -> Dict[str, Route]:
        routes = {}
        for fn, api_info in list_api_info(service):
            methods = api_info.methods
            path = api_info.path

            if existing is not None and path in existing:
                logger.info(f"Duplicated definition of {path}.")
                continue

            if asyncio.iscoroutinefunction(fn):
                logger.error(f"Async function \"{fn.__name__}\" is not supported.")
                continue

            logger.info(f"Serving {path} as {', '.join(methods)}.")

            handler = FlaskHandler(fn, api_info, self)
            routes[path] = Route(api_info=api_info, fn=fn, handler=handler)
        return routes

    def ok(self, body: Union[str, Iterable[str], None], mimetype: str):
        return self.response_class(body, status=200, mimetype=mimetype)

    def error(self, body: str, mimetype=ContentType.plain.value):
        return self.response_class(body, status=500, mimetype=mimetype)

    @api.get("/")
    def index(self, name: str = None):
        if name is None:
            all_api = []
            for route in self.service_routes.values():
                api_info = route.api_info
                all_api.append({"path": api_info.path})
            return all_api

        path = "/" + name
        if path in self.service_routes:
            fn = self.service_routes[path].fn
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
