#!/usr/bin/env python3

__author__ = "xi"

import asyncio
import base64
import uuid
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock
from types import GeneratorType
from typing import Any, Callable, Dict, Generator, Iterable, List, Literal, Optional, Tuple, Union

from flask import Flask, request as flask_request
from pydantic import BaseModel, Field, TypeAdapter

from libentry import json, logger
from libentry.mcp import api
from libentry.mcp.api import APIInfo, list_api_info
from libentry.mcp.types import BlobResourceContents, CallToolRequestParams, CallToolResult, Implementation, \
    InitializeRequestParams, InitializeResult, JSONRPCError, JSONRPCNotification, JSONRPCRequest, JSONRPCResponse, \
    ListResourcesResult, ListToolsResult, MIME, ReadResourceRequestParams, ReadResourceResult, Resource, SSE, \
    ServerCapabilities, SubroutineError, SubroutineResponse, TextContent, TextResourceContents, Tool, ToolSchema, \
    ToolsCapability
from libentry.schema import APISignature, get_api_signature, query_api

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


class SubroutineAdapter:

    def __init__(self, fn: Callable, api_signature: Optional[APISignature] = None):
        self.fn = fn
        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__

        self.api_signature = api_signature or get_api_signature(fn)

    def __call__(
            self,
            request: Any,
            args: Optional[Dict[str, str]] = None
    ) -> Union[SubroutineResponse, Iterable[SubroutineResponse]]:
        if isinstance(request, BaseModel):
            request = request.model_dump()

        if not isinstance(request, Dict):
            raise ValueError(
                f"Subroutine only accepts JSON object as input. "
                f"Expect \"dict\", got \"{type(request)}\"."
            )

        if args is not None:
            conflicts = args.keys() & request.keys()
            if len(conflicts) > 0:
                raise ValueError(f"Duplicated fields: \"{conflicts}\".")
            request = {**args, **request}

        try:
            input_model = self.api_signature.input_model
            if input_model is not None:
                # This is the special case: the only one argument is a BaseModel object.
                # In this case, we omit this argument name, and validate directly.
                arg = input_model.model_validate(request or {})
                response = self.fn(arg)
            else:
                # The arguments are bundled together to perform validation.
                bundled_model = self.api_signature.bundled_model
                kwargs = bundled_model.model_validate(request or {}).model_dump()
                response = self.fn(**kwargs)
        except Exception as e:
            return SubroutineResponse(error=SubroutineError.from_exception(e))

        if not isinstance(response, GeneratorType):
            return SubroutineResponse(result=response)
        else:
            return self._iter_response(response)

    @staticmethod
    def _iter_response(
            results: Iterable[Any]
    ) -> Generator[SubroutineResponse, None, Optional[SubroutineResponse]]:
        it = iter(results)
        try:
            while True:
                result = next(it)
                if not isinstance(result, SubroutineResponse):
                    result = SubroutineResponse(result=result)
                yield result
        except StopIteration as e:
            result = e.value
            if not isinstance(result, SubroutineResponse):
                result = SubroutineResponse(result=result)
            return result
        except Exception as e:
            yield SubroutineResponse(error=SubroutineError.from_exception(e))
        return None


class JSONRPCAdapter:

    def __init__(self, fn: Callable, api_signature: Optional[APISignature] = None):
        self.fn = fn
        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__

        self.api_signature = api_signature or get_api_signature(fn)
        self.type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCNotification])

    def __call__(
            self,
            request: Union[JSONRPCRequest, JSONRPCNotification, Dict[str, Any]],
            args: Optional[Dict[str, str]] = None
    ) -> Union[JSONRPCResponse, Iterable[JSONRPCResponse], None]:
        if isinstance(request, Dict):
            if args is not None:
                conflicts = args.keys() & request.keys()
                if len(conflicts) > 0:
                    raise ValueError(f"Duplicated fields: \"{conflicts}\".")
                request = {**args, **request}
            request = self.type_adapter.validate_python(request)

        if isinstance(request, JSONRPCRequest):
            fn = self._apply_request
        elif isinstance(request, JSONRPCNotification):
            fn = self._apply_notification
        else:
            raise ValueError(
                f"JSONRPC only accepts JSONRPCRequest, JSONRPCNotification as input. "
                f"Expect \"JSONRPCRequest\", \"JSONRPCNotification\" or \"dict\", got \"{type(request)}\"."
            )

        try:
            return fn(request)
        except SystemExit as e:
            raise e
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if isinstance(request, JSONRPCRequest):
                return JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=JSONRPCError.from_exception(e)
                )
            else:
                return None

    def _apply_request(
            self,
            request: JSONRPCRequest
    ) -> Union[JSONRPCResponse, Iterable[JSONRPCResponse]]:
        input_model = self.api_signature.input_model
        if input_model is not None:
            arg = input_model.model_validate(request.params or {})
            result = self.fn(arg)
        else:
            bundled_model = self.api_signature.bundled_model
            kwargs = bundled_model.model_validate(request.params or {}).model_dump()
            result = self.fn(**kwargs)

        if not isinstance(result, (GeneratorType, range)):
            if isinstance(result, JSONRPCResponse):
                response = result
            else:
                response = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    result=result
                )
        else:
            response = self._iter_response(
                results=result,
                request_id=request.id
            )

        return response

    @staticmethod
    def _iter_response(
            results: Iterable[Any],
            request_id: Union[int, str]
    ) -> Generator[JSONRPCResponse, None, Optional[JSONRPCResponse]]:
        it = iter(results)
        while True:
            try:
                result = next(it)
                if not isinstance(result, JSONRPCResponse):
                    result = JSONRPCResponse(
                        jsonrpc="2.0",
                        id=request_id,
                        result=result
                    )
                yield result
            except StopIteration as e:
                final_result = e.value
                if not isinstance(final_result, JSONRPCResponse):
                    final_result = JSONRPCResponse(
                        jsonrpc="2.0",
                        id=request_id,
                        result=final_result
                    )
                break
            except Exception as e:
                final_result = JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request_id,
                    error=JSONRPCError.from_exception(e)
                )
                yield final_result
                break
        return final_result

    def _apply_notification(self, request: JSONRPCNotification) -> None:
        input_model = self.api_signature.input_model
        if input_model is not None:
            arg = input_model.model_validate(request.params or {})
            self.fn(arg)
        else:
            bundled_model = self.api_signature.bundled_model
            kwargs = bundled_model.model_validate(request.params or {}).model_dump()
            self.fn(**kwargs)

        return None


class FlaskHandler:

    def __init__(self, fn: Callable, api_info: APIInfo, app: "FlaskServer"):
        assert hasattr(fn, "__name__")
        self.__name__ = fn.__name__

        self.fn = fn
        self.api_info = api_info
        self.app = app

        self.api_signature = get_api_signature(fn)

        self.subroutine_adapter = SubroutineAdapter(fn, self.api_signature)
        self.jsonrpc_adapter = JSONRPCAdapter(fn, self.api_signature)

        adapter_mapping = {
            api.TAG_ENDPOINT: self.fn,
            "free": self.fn,
            "schema_free": self.fn,
            "schema-free": self.fn,
            api.TAG_JSONRPC: self.jsonrpc_adapter,
            "rpc": self.jsonrpc_adapter,
            "mcp": self.jsonrpc_adapter,
        }
        tag = self.api_info.tag if self.api_info else None
        self.default_adapter = adapter_mapping.get(tag, self.subroutine_adapter)

    def __call__(self):
        args = flask_request.args
        data = flask_request.data
        content_type = flask_request.content_type

        args = {**args}
        if data:
            if (not content_type) or content_type == MIME.json.value:
                raw_request = json.loads(data)
            else:
                return self.app.error(f"Unsupported Content-Type: \"{content_type}\".")
        else:
            raw_request = {}

        ################################################################################
        # Call method as MCP
        ################################################################################
        try:
            mcp_response = self.default_adapter(raw_request, args)
        except Exception as e:
            error = json.dumps(SubroutineError.from_exception(e))
            return self.app.error(error, mimetype=MIME.json.value)

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
        it = iter(events)
        try:
            while True:
                item = next(it)
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
        except StopIteration as e:
            item = e.value
            data = item.data if isinstance(item, SSE) else item
            yield "event:"
            yield "return"
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

    # noinspection PyUnusedLocal
    @api.get("/sse", tag=api.TAG_ENDPOINT)
    def sse(
            self,
            raw_request: Any,
            args: Optional[Dict[str, str]] = None
    ) -> Iterable[SSE]:
        session_id = str(uuid.uuid4())
        queue = Queue(8)
        with self.lock:
            self.sse_dict[session_id] = queue

        def _stream():
            yield SSE(event="endpoint", data=f"/sse/message?sessionId={session_id}")
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

    @api.route("/sse/message", tag=api.TAG_ENDPOINT)
    def sse_message(
            self,
            raw_request: Any,
            args: Optional[Dict[str, str]] = None
    ) -> None:
        if not isinstance(raw_request, Dict):
            raise ValueError(
                f"/message only accepts JSON object as input. "
                f"Expect \"dict\", got \"{type(raw_request)}\"."
            )

        if args is not None:
            conflicts = args.keys() & raw_request.keys()
            if len(conflicts) > 0:
                raise ValueError(f"Duplicated fields: \"{conflicts}\".")
            raw_request = {**args, **raw_request}

        ################################################################################
        # session validation
        ################################################################################
        session_id = raw_request.get("sessionId")
        if session_id is None:
            raise RuntimeError("You should start a session by request the \"/sse\" endpoint first.")
        with self.lock:
            if session_id not in self.sse_dict:
                raise RuntimeError(f"Invalid session: \"{session_id}\".")

        ################################################################################
        # validate request
        ################################################################################
        type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCNotification])
        request = type_adapter.validate_python(raw_request)

        ################################################################################
        # call the mcp method
        ################################################################################
        path = f"/{request.method}"
        route = self.service_routes.get(path, self.builtin_routes.get(path))
        if route is None:
            raise RuntimeError(f"Method {request.method} doesn't exist.")

        response = route.handler.jsonrpc_adapter(request)

        ################################################################################
        # put response
        ################################################################################
        if isinstance(request, JSONRPCNotification):
            return None

        with self.lock:
            queue = self.sse_dict[session_id]

        # todo: remove debug info
        print("/sse/message")
        print(request)
        print(response)
        print()

        if not isinstance(response, (GeneratorType, range)):
            queue.put(response)
        else:
            it = iter(response)
            last_response = None
            while True:
                try:
                    last_response = next(it)
                except StopIteration as e:
                    final_response = e.value
                    break
            if final_response is None:
                final_response = last_response

            if final_response is None:
                raise RuntimeError(f"Method {request.method} doesn't return anything.")

            queue.put(final_response)
        return None


class JSONRPCService:

    def __init__(
            self,
            service_routes: Dict[str, "Route"],
            builtin_routes: Dict[str, "Route"]
    ):
        self.service_routes = service_routes
        self.builtin_routes = builtin_routes

        self.type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCNotification])

    @api.route(tag=api.TAG_ENDPOINT)
    def message(
            self,
            raw_request: Any,
            args: Optional[Dict[str, str]] = None
    ) -> Union[JSONRPCResponse, Iterable[JSONRPCResponse], None]:
        if isinstance(raw_request, List):
            raise ValueError("Batching RCP requests is not supported yet.")

        if not isinstance(raw_request, Dict):
            raise ValueError(
                f"/message only accepts JSON object as input. "
                f"Expect \"dict\", got \"{type(raw_request)}\"."
            )

        if args is not None:
            conflicts = args.keys() & raw_request.keys()
            if len(conflicts) > 0:
                raise ValueError(f"Duplicated fields: \"{conflicts}\".")
            raw_request = {**args, **raw_request}

        request = self.type_adapter.validate_python(raw_request)
        path = f"/{request.method}"
        route = self.service_routes.get(path, self.builtin_routes.get(path))
        if route is None:
            if isinstance(request, JSONRPCRequest):
                return JSONRPCResponse(
                    jsonrpc="2.0",
                    id=request.id,
                    error=JSONRPCError(message=f"Method \"{request.method}\" doesn't exist.")
                )
            else:
                return None
        return route.handler.jsonrpc_adapter(request)


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

    @api.route()
    def ping(self):
        return None


class NotificationsService:

    @api.route("/notifications/initialized")
    def notifications_initialized(self):
        pass


class ToolsService:

    def __init__(self, service_routes: Dict[str, "Route"]):
        self.service_routes = service_routes
        self._tool_routes = None

    def get_tool_routes(self) -> Dict[str, "Route"]:
        if self._tool_routes is None:
            self._tool_routes = {}
            for route in self.service_routes.values():
                api_info = route.api_info
                if api_info.tag != api.TAG_TOOL:
                    continue
                self._tool_routes[api_info.name] = route
        return self._tool_routes

    @api.route("/tools/list")
    def tools_list(self) -> ListToolsResult:
        tools = []
        for name, route in self.get_tool_routes().items():
            api_info = route.api_info
            api_models = route.fn if isinstance(route.fn, APISignature) else get_api_signature(route.fn)
            args_model = api_models.input_model or api_models.bundled_model
            tool = Tool(
                name=api_info.name,
                description=api_info.description,
                inputSchema=ToolSchema.model_validate(args_model.model_json_schema())
            )
            tools.append(tool)
            # schema = query_api(route.fn)
            # input_schema = schema.context[schema.input_schema]
            # for field in input_schema.fields:
            #     type_ = field.type
            #     if isinstance(type_, List):
            #         type_ = "|".join(type_)
            #     tool.inputSchema.properties[field.name] = ToolProperty(
            #         type=type_,
            #         description=field.description
            #     )
            #     if field.is_required:
            #         tool.inputSchema.required.append(field.name)
        return ListToolsResult(tools=tools)

    @api.route("/tools/call")
    def tools_call(self, params: CallToolRequestParams) -> Union[CallToolResult, Iterable[CallToolResult]]:
        route = self.get_tool_routes().get(params.name)
        if route is None:
            raise RuntimeError(f"Tool \"{params.name}\" doesn't exist.")

        try:
            response = route.handler.subroutine_adapter(params.arguments)
        except Exception as e:
            error = json.dumps(SubroutineError.from_exception(e))
            return CallToolResult(
                content=[TextContent(text=error)],
                isError=True
            )

        if not isinstance(response, GeneratorType):
            if response.error is not None:
                text = json.dumps(response.error)
                return CallToolResult(
                    content=[TextContent(text=text)],
                    isError=True
                )
            else:
                result = response.result
                text = json.dumps(result) if isinstance(result, (Dict, BaseModel)) else str(result)
                return CallToolResult(
                    content=[TextContent(text=text)],
                    isError=False
                )
        else:
            return self._iter_tool_results(response)

    @staticmethod
    def _iter_tool_results(
            responses: Iterable[SubroutineResponse]
    ) -> Generator[CallToolResult, None, Optional[CallToolResult]]:
        try:
            it = iter(responses)
            while True:
                response = next(it)
                if response.error is not None:
                    text = json.dumps(response.error)
                    yield CallToolResult(
                        content=[TextContent(text=text)],
                        isError=True
                    )
                    break
                else:
                    result = response.result
                    text = json.dumps(result) if isinstance(result, (Dict, BaseModel)) else str(result)
                    yield CallToolResult(
                        content=[TextContent(text=text)],
                        isError=False
                    )
        except StopIteration as e:
            response = e.value
            if response is not None:
                if response.error is not None:
                    text = json.dumps(response.error)
                    return CallToolResult(
                        content=[TextContent(text=text)],
                        isError=True
                    )
                else:
                    result = response.result
                    text = json.dumps(result) if isinstance(result, (Dict, BaseModel)) else str(result)
                    return CallToolResult(
                        content=[TextContent(text=text)],
                        isError=False
                    )
        except Exception as e:
            text = json.dumps(SubroutineError.from_exception(e))
            error = CallToolResult(
                content=[TextContent(text=text)],
                isError=True
            )
            yield error
        return None


class ResourcesService:

    def __init__(self, service_routes: Dict[str, "Route"]):
        self.service_routes = service_routes

        self._resource_routes = None

    def get_resource_routes(self) -> Dict[str, "Route"]:
        if self._resource_routes is None:
            self._resource_routes = {}
            for route in self.service_routes.values():
                api_info = route.api_info
                if api_info.tag != api.TAG_RESOURCE:
                    continue
                uri = api_info.model_extra.get("uri", api_info.path)
                self._resource_routes[uri] = route
        return self._resource_routes

    @api.route("/resources/list")
    def resources_list(self) -> ListResourcesResult:
        resources = []
        for uri, route in self.get_resource_routes().items():
            api_info = route.api_info
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
        route = self.get_resource_routes().get(request.uri)
        if route is None:
            raise RuntimeError(f"Resource \"{request.uri}\" doesn't exist.")

        api_info = route.api_info
        result = ReadResourceResult(contents=[])
        content = route.fn()
        mime_type = api_info.model_extra.get("mimeType")
        if not isinstance(content, GeneratorType):
            if isinstance(content, str):
                result.contents.append(TextResourceContents(
                    uri=request.uri,
                    mimeType=mime_type or "text/*",
                    text=content
                ))
            elif isinstance(content, bytes):
                result.contents.append(BlobResourceContents(
                    uri=request.uri,
                    mimeType=mime_type or "binary/*",
                    blob=base64.b64encode(content).decode()
                ))
            else:
                raise RuntimeError(f"Unsupported content type \"{type(content)}\".")
            return result
        else:
            # todo: this branch is not tested yet
            return self._iter_resource_results(content, request.uri, mime_type)

    @staticmethod
    def _iter_resource_results(
            contents: Iterable[Union[str, bytes]],
            uri: str,
            mime_type: Optional[str] = None
    ) -> Generator[ReadResourceResult, None, Optional[ReadResourceResult]]:
        # todo: returned content need to be processed
        for content in contents:
            if isinstance(content, str):
                yield ReadResourceResult(contents=[TextResourceContents(
                    uri=uri,
                    mimeType=mime_type or "text/*",
                    text=content
                )])
            elif isinstance(content, bytes):
                yield ReadResourceResult(contents=[BlobResourceContents(
                    uri=uri,
                    mimeType=mime_type or "binary/*",
                    blob=base64.b64encode(content).decode()
                )])
            else:
                raise RuntimeError(f"Unsupported content type \"{type(content)}\".")


@dataclass
class Route:
    api_info: APIInfo
    fn: Callable
    handler: FlaskHandler


class FlaskServer(Flask):

    def __init__(self, service, options: Dict[str, Any]):
        super().__init__(__name__)
        self.options = options
        self.access_control_allow_origin = self.options.get("access_control_allow_origin")
        self.access_control_allow_methods = self.options.get("access_control_allow_methods")

        self.service_routes = {}
        self.builtin_routes = {}

        self.service = service
        self.builtin_services = [
            SSEService(self.service_routes, self.builtin_routes),
            JSONRPCService(self.service_routes, self.builtin_routes),
            LifeCycleService(),
            NotificationsService(),
            ToolsService(self.service_routes),
            ResourcesService(self.service_routes)
        ]

        logger.info("Initializing Flask application.")
        existing_routes = {}

        input_services = self.service if isinstance(self.service, (List, Tuple)) else [self.service]
        for input_service in input_services:
            routes = self._create_routes(input_service)
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

            handler = FlaskHandler(fn, api_info, self)
            routes[path] = Route(api_info=api_info, fn=fn, handler=handler)

            mode = api.TAG_ENDPOINT
            if isinstance(handler.default_adapter, SubroutineAdapter):
                mode = api.TAG_SUBROUTINE
            elif isinstance(handler.default_adapter, JSONRPCAdapter):
                mode = api.TAG_JSONRPC
            logger.info(f"{mode.capitalize()}:\tmethod={'|'.join(methods)}\tpath={path}")
        return routes

    def ok(self, body: Union[str, Iterable[str], None], mimetype: str):
        response = self.response_class(body, status=200, mimetype=mimetype)
        if self.access_control_allow_origin:
            response.headers["Access-Control-Allow-Origin"] = "*"
        if self.access_control_allow_methods:
            response.headers["Access-Control-Allow-Methods"] = "GET, POST"
        return response

    def error(self, body: str, mimetype=MIME.plain.value):
        response = self.response_class(body, status=500, mimetype=mimetype)
        if self.access_control_allow_origin:
            response.headers["Access-Control-Allow-Origin"] = "*"
        if self.access_control_allow_methods:
            response.headers["Access-Control-Allow-Methods"] = "GET, POST"
        return response

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
        if isinstance(self.service_type, (List, Tuple)):
            if isinstance(self.service_config, (List, Tuple)) and len(self.service_config) == len(self.service_type):
                service = [
                    self._create_service(t, c)
                    for t, c in zip(self.service_type, self.service_config)
                ]
            elif self.service_config is None:
                service = [
                    self._create_service(t, self.service_config)
                    for t in self.service_type
                ]
            else:
                raise RuntimeError(
                    f"You are going to run {len(self.service_type)} services. "
                    "In the multiple service mode, every `service_type` requires a `service_config` object."
                )
        else:
            service = self._create_service(self.service_type, self.service_config)
        logger.info("Service initialized.")

        return FlaskServer(service, self.options)

    @staticmethod
    def _create_service(service_type, service_config):
        if isinstance(service_type, type) or callable(service_type):
            service = service_type(service_config) if service_config else service_type()
        elif service_config is None:
            logger.warning(
                "Be careful! It is not recommended to start the server from a service instance. "
                "Use service_type and service_config instead."
            )
            service = service_type
        else:
            raise TypeError(f"Invalid service type \"{type(service_type)}\".")
        return service


class RunServiceConfig(BaseModel):
    """Run service config."""

    host: str = Field(
        title="Hostname",
        description=(
            "The hostname of the server that runs the service. "
            "IP address or domain name."
        ),
        default="0.0.0.0"
    )
    port: int = Field(
        title="Port number",
        description="The port that the service listened to.",
    )
    num_workers: int = Field(
        title="Number of workers",
        description="The number of workers (processes) to run the service.",
        default=1
    )
    num_threads: int = Field(
        title="Number of threads",
        description="The number of threads for each worker.",
        default=20
    )
    num_connections: Optional[int] = Field(
        title="Max number of connections",
        description="The maximum number of simultaneous clients (or socket connections).",
        default=1024
    )
    backlog: Optional[int] = Field(
        title="The maximum number of pending connections",
        description="The maximum number of pending connections.",
        default=2048
    )
    worker_class: Literal["sync", "gthread", "eventlet", "gevent", "tornado"] = Field(
        title="The type of workers to use",
        description="The type of workers to use.",
        default="gthread"
    )
    timeout: int = Field(
        title="Worker timeout",
        description="Workers silent for more than this many seconds are killed and restarted.",
        default=60
    )
    keyfile: Optional[str] = Field(
        title="SSL key file",
        description="SSL key file.",
        default=None
    )
    keyfile_password: Optional[str] = Field(
        title="SSL key file password",
        description="SSL key file password.",
        default=None
    )
    certfile: Optional[str] = Field(
        title="SSL certificate file",
        description="SSL certificate file.",
        default=None
    )
    access_control_allow_origin: Optional[str] = Field(
        title="Access control allow origin",
        description="Access control allow origin.",
        default="*"
    )
    access_control_allow_methods: Optional[str] = Field(
        title="Access control allow methods",
        description="Access control allow methods.",
        default="GET, POST"
    )
    name: Optional[str] = Field(
        title="服务实例名称",
        description="服务实例名称，会在进程命令行中显示。",
        default=None
    )


def run_service(
        service_type: Union[Union[type, Callable], List, Tuple],
        service_config=None,
        run_config: Optional[RunServiceConfig] = None,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        num_workers: Optional[int] = None,
        num_threads: Optional[int] = None,
        num_connections: Optional[int] = None,
        backlog: Optional[int] = None,
        worker_class: Optional[str] = None,
        timeout: Optional[int] = None,
        keyfile: Optional[str] = None,
        keyfile_password: Optional[str] = None,
        certfile: Optional[str] = None
) -> None:
    kwargs = {}
    if host is not None:
        kwargs["host"] = host
    if port is not None:
        kwargs["port"] = port
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    if num_threads is not None:
        kwargs["num_threads"] = num_threads
    if num_connections is not None:
        kwargs["num_connections"] = num_connections
    if backlog is not None:
        kwargs["backlog"] = backlog
    if worker_class is not None:
        kwargs["worker_class"] = worker_class
    if timeout is not None:
        kwargs["timeout"] = timeout
    if keyfile is not None:
        kwargs["keyfile"] = keyfile
    if keyfile_password is not None:
        kwargs["keyfile_password"] = keyfile_password
    if certfile is not None:
        kwargs["certfile"] = certfile

    if run_config is None:
        run_config = RunServiceConfig(**kwargs)
    else:
        for name, value in kwargs.items():
            setattr(run_config, name, value)

    if run_config.name is None:
        name = service_type.__name__
        module = service_type.__module__
        if module != "builtins":
            name = f"{module}.{name}"
        run_config.name = name
    run_config.name = f"{run_config.name} ({run_config.host}:{run_config.port})"

    logger.info("Starting gunicorn server.")

    def ssl_context(config, _default_ssl_context_factory):
        import ssl
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(
            certfile=config.certfile,
            keyfile=config.keyfile,
            password=run_config.keyfile_password
        )
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        return context

    options = {
        "bind": f"{run_config.host}:{run_config.port}",
        "workers": run_config.num_workers,
        "threads": run_config.num_threads,
        "timeout": run_config.timeout,
        "worker_connections": run_config.num_connections,
        "backlog": run_config.backlog,
        "keyfile": run_config.keyfile,
        "certfile": run_config.certfile,
        "worker_class": run_config.worker_class,
        "ssl_context": ssl_context,
        "access_control_allow_origin": run_config.access_control_allow_origin,
        "access_control_allow_methods": run_config.access_control_allow_methods,
        "proc_name": run_config.name,
    }
    for name, value in options.items():
        logger.info(f"Option {name}: {value}")
    GunicornApplication(service_type, service_config, options).run()
