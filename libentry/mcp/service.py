#!/usr/bin/env python3

__author__ = "xi"

import asyncio
import base64
import uuid
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
from types import GeneratorType
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Type, Union

from flask import Flask, request as flask_request
from pydantic import BaseModel, TypeAdapter

from libentry import json, logger
from libentry.mcp import api
from libentry.mcp.api import APIInfo, list_api_info
from libentry.mcp.types import BlobResourceContents, CallToolRequestParams, CallToolResult, Implementation, \
    InitializeRequestParams, \
    InitializeResult, JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse, ListResourcesResult, \
    ListToolsResult, MIME, ReadResourceRequestParams, ReadResourceResult, Resource, SSE, \
    ServerCapabilities, TextContent, TextResourceContents, Tool, ToolProperty, ToolSchema, ToolsCapability
from libentry.schema import get_api_signature, query_api

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

        self.api_signature = get_api_signature(fn)

        self.type_adapter = TypeAdapter(
            Union[
                JSONRPCRequest,
                JSONRPCNotification,
                Dict[str, Any]
            ]  # todo: need a fallback type
        )

    def __call__(
            self,
            request: Union[JSONRPCRequest, JSONRPCNotification, Dict[str, Any]]
    ) -> Union[JSONRPCResponse, Iterable[JSONRPCResponse], Dict[str, Any], Iterable[Dict[str, Any]], None]:
        request = self.type_adapter.validate_python(request)

        if isinstance(request, (JSONRPCRequest, JSONRPCNotification)):
            try:
                return self._call_as_jsonrpc(request)
            except SystemExit as e:
                raise e
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                if isinstance(request, JSONRPCNotification):
                    return None
                return JSONRPCResponse(
                    id=request.id,
                    error=JSONRPCError.from_exception(e)
                )
        else:
            return self._call_as_fallback(request)

    def _call_as_jsonrpc(
            self,
            request: Union[JSONRPCRequest, JSONRPCNotification]
    ) -> Union[JSONRPCResponse, Iterable[JSONRPCResponse], None]:
        input_model = self.api_signature.input_model
        if input_model is not None:
            if issubclass(input_model, (JSONRPCResponse, JSONRPCNotification)):
                arg = input_model.model_validate(request.model_dump())
                result = self.fn(arg)
            else:
                arg = input_model.model_validate(request.params or {})
                result = self.fn(arg)
        else:
            bundled_model = self.api_signature.bundled_model
            kwargs = bundled_model.model_validate(request.params or {}).model_dump()
            result = self.fn(**kwargs)

        if isinstance(request, JSONRPCNotification):
            return None

        if not isinstance(result, (GeneratorType, range)):
            return (
                result if isinstance(result, JSONRPCResponse) else
                JSONRPCResponse(id=request.id, result=result)
            )
        else:
            return self._iter_jsonrpc_response(request, result)

    @staticmethod
    def _iter_jsonrpc_response(
            request: JSONRPCRequest,
            results: Iterable[Any]
    ) -> Generator[JSONRPCResponse, None, Optional[JSONRPCResponse]]:
        it = iter(results)
        while True:
            try:
                result = next(it)
                if not isinstance(result, JSONRPCResponse):
                    result = JSONRPCResponse(
                        jsonrpc="2.0",
                        id=request.id,
                        result=result
                    )
                yield result
            except StopIteration as e:
                final_result = e.value
                if not isinstance(final_result, JSONRPCResponse):
                    final_result = JSONRPCResponse(
                        jsonrpc="2.0",
                        id=request.id,
                        result=final_result
                    )
                return final_result

    def _call_as_fallback(self, request: Union[Dict[str, Any], BaseModel]):
        if isinstance(request, BaseModel):
            request = request.model_dump()

        input_model = self.api_signature.input_model
        if input_model is not None:
            arg = input_model.model_validate(request or {})
            return self.fn(arg)
        else:
            bundled_model = self.api_signature.bundled_model
            kwargs = bundled_model.model_validate(request or {}).model_dump()
            return self.fn(**kwargs)

        # if issubclass(self.input_schema, (JSONRPCResponse, JSONRPCNotification)):
        #     raw_params = self.input_schema.model_validate(request.model_dump())
        #     return self.fn(raw_params)
        # elif issubclass(self.input_schema, BaseModel):
        #     # Note that "input_schema is not None" means:
        #     # (1) The function has only one argument;
        #     # (2) The arguments is a BaseModel.
        #     # In this case, the request data can be directly validated as a "BaseModel" and
        #     # subsequently passed to the function as a single object.
        #     pydantic_params = self.input_schema.model_validate(request.params or {})
        #     return self.fn(pydantic_params)
        # else:
        #     # The function has multiple arguments, and the request data bundle them as a single object.
        #     # So, they should be unpacked before pass to the function.
        #     kwargs = request.params or {}
        #     return self.fn(**kwargs)


class FlaskHandler:

    def __init__(self, fn: Callable, api_info: APIInfo, app: "FlaskServer"):
        self.fn = fn if api_info.tag == "schema_free" else JSONRPCAdapter(fn)
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
            if (not content_type) or content_type == MIME.json.value:
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
        mimetype = MIME.json.value if MIME.json.value in accepts else MIME.plain.value
        if mcp_response is None:
            return self.app.ok(
                None,
                mimetype=mimetype
            )
        elif isinstance(mcp_response, BaseModel):
            # BaseModel
            return self.app.ok(
                json.dumps(mcp_response.model_dump(exclude_none=True)),
                mimetype=mimetype
            )
        elif isinstance(mcp_response, (Dict, List)):
            # JSON Object and Array
            return self.app.ok(
                json.dumps(mcp_response),
                mimetype=mimetype
            )
        elif isinstance(mcp_response, (GeneratorType, range)):
            # Stream response
            if MIME.sse.value in accepts:
                # SSE is first considered
                return self.app.ok(
                    self._iter_sse_stream(mcp_response),
                    mimetype=MIME.sse.value
                )
            else:
                # JSON Stream for fallback
                return self.app.ok(
                    self._iter_sse_stream(mcp_response),
                    mimetype=MIME.json_stream.value
                )
        else:
            # Plain text
            return self.app.ok(
                str(mcp_response),
                mimetype=MIME.plain.value
            )

    def _iter_sse_stream(self, events: Iterable[Union[SSE, Dict[str, Any]]]) -> Iterable[str]:
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
                    # BaseModel
                    yield json.dumps(data.model_dump(exclude_none=True))
                elif isinstance(data, (Dict, List)):
                    # JSON Object and Array
                    yield json.dumps(data)
                else:
                    # Plain text
                    yield str(data)
            yield "\n\n"

    def _iter_json_stream(self, objs: Iterable[Union[BaseModel, Dict[str, Any]]]) -> Iterable[str]:
        for obj in objs:
            if isinstance(obj, BaseModel):
                # BaseModel
                yield json.dumps(obj.model_dump(exclude_none=True))
            elif isinstance(obj, (Dict, List)):
                # JSON Object and Array
                yield json.dumps(obj)
            else:
                # Plain text
                yield str(obj)
            yield "\n"


class SSEService:

    def __init__(
            self,
            service_routes: Dict[str, "Route"],
            builtin_routes: Dict[str, "Route"]
    ):
        self.service_routes = service_routes
        self.builtin_routes = builtin_routes
        self.lock = Lock()
        self.sse_dict = {}

    @api.get("/sse", tag="schema_free")
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
                        ping_request = JSONRPCRequest(jsonrpc="2.0", id=str(uuid.uuid4()), method="ping")
                        yield SSE(event="message", data=ping_request)
            finally:
                with self.lock:
                    del self.sse_dict[session_id]
                logger.info(f"Session {session_id} cleaned.")

        return _stream()

    @api.route("/message", tag="schema_free")
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

        ################################################################################
        # call the mcp method
        ################################################################################
        path = "/" + mcp_request.method
        route = self.service_routes.get(path, self.builtin_routes.get(path))
        if route is None:
            raise RuntimeError(f"Method {mcp_request.method} doesn't exist.")

        response = route.handler.fn(mcp_request)

        ################################################################################
        # put response
        ################################################################################
        if isinstance(mcp_request, JSONRPCNotification):
            return None

        with self.lock:
            queue = self.sse_dict[session_id]

        # todo: remove debug info
        print("/message")
        print(mcp_request)
        print(response)
        print()

        if not isinstance(response, (GeneratorType, range)):
            queue.put(response)
        else:
            it = iter(response)
            while True:
                try:
                    next(it)
                except StopIteration as e:
                    final_response = e.value
                    break
            if final_response is None:
                # todo: is it better to return a JSONRPCResponse with error?
                raise RuntimeError("Stream output is not supported.")

            queue.put(final_response)
        return None


class LifeCycleService:

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


class NotificationsService:

    @api.route("/notifications/initialized")
    def notifications_initialized(self):
        pass


class ToolsService:

    def __init__(self, service_routes: Dict[str, "Route"]):
        self.service_routes = service_routes

    @api.route("/tools/list")
    def tools_list(self) -> ListToolsResult:
        tools = []
        for route in self.service_routes.values():
            api_info = route.api_info
            if api_info.tag != "tool":
                continue
            tool = Tool(
                name=api_info.name,
                description=api_info.description,
                inputSchema=ToolSchema()
            )
            tools.append(tool)
            schema = query_api(route.fn)
            input_schema = schema.context[schema.input_schema]
            for field in input_schema.fields:
                type_ = field.type
                if isinstance(type_, List):
                    type_ = "|".join(type_)
                tool.inputSchema.properties[field.name] = ToolProperty(
                    type=type_,
                    description=field.description
                )
                if field.is_required:
                    tool.inputSchema.required.append(field.name)
        return ListToolsResult(tools=tools)

    @api.route("/tools/call")
    def tools_call(self, params: CallToolRequestParams) -> Union[CallToolResult, Iterable[CallToolResult]]:
        route = self.service_routes.get(f"/{params.name}")
        if route is None:
            return CallToolResult(
                content=[TextContent(text=f"Tool \"{params.name}\" doesn't exist.")],
                isError=True
            )
        if route.api_info.tag != "tool":
            return CallToolResult(
                content=[TextContent(text=f"\"{params.name}\" is not defined as a tool.")],
                isError=True
            )

        rpc_request = JSONRPCRequest(
            jsonrpc="2.0",
            id=str(uuid.uuid4()),
            method=params.name,
            params=params.arguments
        )
        rpc_response = route.handler.fn(rpc_request)

        if not isinstance(rpc_response, GeneratorType):
            if rpc_response.error is not None:
                error = rpc_response.error
                text = json.dumps(error) if isinstance(error, (Dict, BaseModel)) else str(error)
                return CallToolResult(
                    content=[TextContent(text=text)],
                    isError=True
                )
            else:
                result = rpc_response.result
                text = json.dumps(result) if isinstance(result, (Dict, BaseModel)) else str(result)
                return CallToolResult(
                    content=[TextContent(text=text)],
                    isError=False
                )
        else:
            return self._iter_tool_result(rpc_response)

    @staticmethod
    def _iter_tool_result(
            response: Iterable[JSONRPCResponse]
    ) -> Generator[CallToolResult, None, CallToolResult]:
        text_list = []
        # todo: add an error list?
        for item in response:
            if item.error is not None:
                error = item.error
                text = json.dumps(error) if isinstance(error, (Dict, BaseModel)) else str(error)
                yield CallToolResult(
                    content=[TextContent(text=text)],
                    isError=True
                )
            else:
                result = item.result
                text = json.dumps(result) if isinstance(result, (Dict, BaseModel)) else str(result)
                text_list.append(text)
                yield CallToolResult(
                    content=[TextContent(text=text)],
                    isError=False
                )
        return CallToolResult(
            content=[TextContent(text="".join(text_list))],
            isError=False
        )


class ResourcesService:

    def __init__(self, service_routes: Dict[str, "Route"]):
        self.service_routes = service_routes

    @api.route("/resources/list")
    def resources_list(self) -> ListResourcesResult:
        resources = []
        for route in self.service_routes.values():
            api_info = route.api_info
            if api_info.tag != "resource":
                continue
            uri = api_info.model_extra.get("uri", api_info.path)
            resources.append(Resource(
                uri=uri,
                name=api_info.name,
                description=api_info.description,
                mimeType=api_info.model_extra.get("mimeType"),
                size=api_info.model_extra.get("size")
            ))
        return ListResourcesResult(resources=resources)

    @api.route("/resources/read")
    def resources_read(
            self,
            request: ReadResourceRequestParams
    ) -> Union[ReadResourceResult, Iterable[ReadResourceResult]]:
        # todo: optimize the way of finding the resource route
        for route in self.service_routes.values():
            api_info = route.api_info
            if api_info.tag != "resource":
                continue
            uri = api_info.model_extra.get("uri")
            if uri != request.uri:
                continue
            result = ReadResourceResult(contents=[])
            content = route.fn()
            if not isinstance(content, GeneratorType):
                if isinstance(content, str):
                    mime_type = api_info.model_extra.get("mimeType", "text/*")
                    result.contents.append(TextResourceContents(
                        uri=uri,
                        mimeType=mime_type,
                        text=content
                    ))
                elif isinstance(content, bytes):
                    mime_type = api_info.model_extra.get("mimeType", "binary/*")
                    result.contents.append(BlobResourceContents(
                        uri=uri,
                        mimeType=mime_type,
                        blob=base64.b64encode(content).decode()
                    ))
                else:
                    raise RuntimeError(f"Unsupported content type \"{type(content)}\".")
                return result
            else:
                # todo: this branch is not tested yet
                return self._iter_resource_result(content, uri, api_info)

        raise RuntimeError(f"Resource \"{request.uri}\" doesn't exist.")

    @staticmethod
    def _iter_resource_result(
            contents: Iterable[Union[str, bytes]],
            uri: str,
            api_info: APIInfo
    ) -> Generator[ReadResourceResult, None, Optional[ReadResourceResult]]:
        final_results = []
        for content in contents:
            if isinstance(content, str):
                mime_type = api_info.model_extra.get("mimeType", "text/*")
                yield ReadResourceResult(contents=[TextResourceContents(
                    uri=uri,
                    mimeType=mime_type,
                    text=content
                )])
                final_results.append(content)
            elif isinstance(content, bytes):
                mime_type = api_info.model_extra.get("mimeType", "binary/*")
                yield ReadResourceResult(contents=[BlobResourceContents(
                    uri=uri,
                    mimeType=mime_type,
                    blob=base64.b64encode(content).decode()
                )])
                final_results.append(content)
            else:
                raise RuntimeError(f"Unsupported content type \"{type(content)}\".")

        if len(final_results) == 0:
            return None

        if isinstance(final_results[0], str):
            mime_type = api_info.model_extra.get("mimeType", "text/*")
            return ReadResourceResult(contents=[TextResourceContents(
                uri=uri,
                mimeType=mime_type,
                text="".join(final_results)
            )])
        else:
            mime_type = api_info.model_extra.get("mimeType", "binary/*")
            return ReadResourceResult(contents=[BlobResourceContents(
                uri=uri,
                mimeType=mime_type,
                blob=base64.b64encode(b"".join(final_results)).decode()
            )])


@dataclass
class Route:
    api_info: APIInfo
    fn: Callable
    handler: FlaskHandler


class FlaskServer(Flask):

    def __init__(self, service):
        super().__init__(__name__)

        self.service_routes = {}
        self.builtin_routes = {}

        self.service = service
        self.builtin_services = [
            SSEService(self.service_routes, self.builtin_routes),
            LifeCycleService(),
            NotificationsService(),
            ToolsService(self.service_routes),
            ResourcesService(self.service_routes)
        ]

        logger.info("Initializing Flask application.")
        existing_routes = {}

        routes = self._create_routes(self.service)
        self.service_routes.update(routes)
        existing_routes.update(self.service_routes)

        for builtin_service in self.builtin_services:
            routes = self._create_routes(builtin_service, existing_routes)
            self.builtin_routes.update(routes)
            existing_routes.update(routes)

        routes = self._create_routes(self, existing_routes)
        self.builtin_routes.update(routes)
        existing_routes.update(routes)

        for route in (*self.service_routes.values(), *self.builtin_routes.values()):
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

    def error(self, body: str, mimetype=MIME.plain.value):
        return self.response_class(body, status=500, mimetype=mimetype)

    @api.get("/")
    def index(self, name: str = None):
        if name is None:
            all_api = []
            for route in self.service_routes.values():
                api_info = route.api_info
                all_api.append(api_info)
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
