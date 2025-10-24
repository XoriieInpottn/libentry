#!/usr/bin/env python3

__author__ = "xi"

import abc
import uuid
from queue import Queue
from threading import Semaphore, Thread
from time import sleep
from types import GeneratorType
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Type, Union
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, TypeAdapter

from libentry import json
from libentry.mcp.api import HasRequestPath
from libentry.mcp.types import CallToolRequestParams, CallToolResult, ClientCapabilities, HTTPOptions, HTTPRequest, \
    HTTPResponse, Implementation, InitializeRequestParams, InitializeResult, JSONObject, JSONRPCError, \
    JSONRPCNotification, JSONRPCRequest, JSONRPCResponse, JSONType, ListResourcesResult, ListToolsResult, MIME, \
    ReadResourceRequestParams, ReadResourceResult, SSE, SubroutineError, SubroutineResponse


class ServiceError(RuntimeError):

    def __init__(
            self,
            message: str,
            cause: Optional[str] = None,
            _traceback: Optional[str] = None
    ):
        self.message = message
        self.cause = cause
        self.traceback = _traceback

    def __str__(self):
        lines = []
        if self.message:
            lines += [self.message, "\n\n"]
        if self.cause:
            lines += ["This is caused by server side error ", self.cause, ".\n"]
        if self.traceback:
            lines += ["Below is the stacktrace:\n", self.traceback.rstrip()]
        return "".join(lines)

    @staticmethod
    def from_subroutine_error(error: SubroutineError):
        return ServiceError(error.message, error.error, error.traceback)

    @staticmethod
    def from_jsonrpc_error(error: JSONRPCError):
        cause = None
        traceback_ = None
        if isinstance(error.data, Dict):
            cause = error.data.get("error")
            traceback_ = error.data.get("traceback")
        return ServiceError(error.message, cause, traceback_)


class SSEDecoder:

    def __init__(self) -> None:
        self._event = ""
        self._data: List[str] = []
        self._last_event_id = ""
        self._retry: Optional[int] = None

    def decode(self, line: str) -> Optional[SSE]:
        if not line:
            if (
                    not self._event
                    and not self._data
                    and not self._last_event_id
                    and self._retry is None
            ):
                return None

            sse = SSE(
                event=self._event,
                data="\n".join(self._data),
            )

            # NOTE: as per the SSE spec, do not reset last_event_id.
            self._event = ""
            self._data = []
            self._retry = None

            return sse

        if line.startswith(":"):
            return None

        fieldname, _, value = line.partition(":")

        if value.startswith(" "):
            value = value[1:]

        if fieldname == "event":
            self._event = value
        elif fieldname == "data":
            self._data.append(value)
        elif fieldname == "id":
            if "\0" in value:
                pass
            else:
                self._last_event_id = value
        elif fieldname == "retry":
            try:
                self._retry = int(value)
            except (TypeError, ValueError):
                pass
        else:
            pass  # Field is ignored.

        return None


class SubroutineMixIn(abc.ABC):

    @abc.abstractmethod
    def subroutine_request(
            self,
            path: Union[str, Type[HasRequestPath], HasRequestPath, Any],
            params: Optional[Union[JSONObject, BaseModel]] = None,
            options: Optional[HTTPOptions] = None
    ) -> Union[SubroutineResponse, Generator[SubroutineResponse, None, None]]:
        raise NotImplementedError()

    def request(
            self,
            path: Union[str, Type[HasRequestPath], HasRequestPath, Any],
            params: Optional[Union[JSONObject, BaseModel]] = None,
            options: Optional[HTTPOptions] = None
    ) -> Union[JSONType, Generator[JSONType, None, None]]:
        response = self.subroutine_request(path, params, options)
        if not isinstance(response, GeneratorType):
            if response.error is None:
                return response.result
            else:
                raise ServiceError.from_subroutine_error(response.error)
        else:
            return self._iter_results_from_subroutine(response)

    @staticmethod
    def _iter_results_from_subroutine(response: Iterable[SubroutineResponse]) -> Generator[JSONType, None, None]:
        it = iter(response)
        try:
            while True:
                chunk = next(it)
                if chunk.error is None:
                    yield chunk.result
                else:
                    raise ServiceError.from_subroutine_error(chunk.error)
        except StopIteration as e:
            chunk = e.value
            if chunk is None:
                return None
            elif chunk.error is None:
                return chunk.result
            else:
                raise ServiceError.from_subroutine_error(chunk.error)

    def get(
            self,
            path: Union[str, Type[HasRequestPath], HasRequestPath, Any],
            options: Optional[HTTPOptions] = None
    ) -> Union[JSONType, Generator[JSONType, None, None]]:
        if options is None:
            options = HTTPOptions(method="GET")
        if options.method != "GET":
            raise ValueError("options.method should be \"GET\".")
        return self.request(path=path, params=None, options=options)

    def post(
            self,
            path: Union[str, Type[HasRequestPath], HasRequestPath, Any],
            params: Optional[Union[JSONObject, BaseModel]] = None,
            options: Optional[HTTPOptions] = None
    ) -> Union[JSONType, Generator[JSONType, None, None]]:
        if options is None:
            options = HTTPOptions(method="POST")
        if options.method != "POST":
            raise ValueError("options.method should be \"POST\".")
        return self.request(path=path, params=params, options=options)


class JSONRPCMixIn(abc.ABC):

    @abc.abstractmethod
    def jsonrpc_request(
            self,
            request: JSONRPCRequest,
            path: Optional[str] = None,
            options: Optional[HTTPOptions] = None
    ) -> Union[JSONRPCResponse, Generator[JSONRPCResponse, None, None]]:
        raise NotImplementedError()

    @abc.abstractmethod
    def jsonrpc_notify(
            self,
            request: JSONRPCNotification,
            path: Optional[str] = None,
            options: Optional[HTTPOptions] = None
    ) -> None:
        raise NotImplementedError()

    def call(
            self,
            method: str,
            params: Optional[JSONObject] = None,
            options: Optional[HTTPOptions] = None
    ) -> Union[JSONType, Generator[JSONType, None, None]]:
        request = JSONRPCRequest(
            jsonrpc="2.0",
            id=str(uuid.uuid4()),
            method=method,
            params=params
        )

        response = self.jsonrpc_request(request, options=options)

        if not isinstance(response, GeneratorType):
            if response.error is None:
                return response.result
            else:
                raise ServiceError.from_jsonrpc_error(response.error)
        else:
            return self._iter_results_from_jsonrpc(response)

    @staticmethod
    def _iter_results_from_jsonrpc(response: Iterable[JSONRPCResponse]) -> Generator[JSONType, None, None]:
        it = iter(response)
        try:
            while True:
                chunk = next(it)
                if chunk.error is None:
                    yield chunk.result
                else:
                    raise ServiceError.from_jsonrpc_error(chunk.error)
        except StopIteration as e:
            chunk = e.value
            if chunk is None:
                return None
            elif chunk.error is None:
                return chunk.result
            else:
                raise ServiceError.from_jsonrpc_error(chunk.error)


class MCPMixIn(JSONRPCMixIn, abc.ABC):

    def initialize(self) -> InitializeResult:
        params = InitializeRequestParams(
            protocolVersion="2024-11-05",
            capabilities=ClientCapabilities(),
            clientInfo=Implementation(name="libentry-client", version="1.0.0")
        ).model_dump(exclude_none=True)

        result = self.call("initialize", params)

        self.jsonrpc_notify(JSONRPCNotification(
            jsonrpc="2.0", method="notifications/initialized"
        ))

        return InitializeResult.model_validate(result)

    def ping(self):
        return self.call("ping")

    def list_tools(self) -> ListToolsResult:
        result = self.call("tools/list")
        return ListToolsResult.model_validate(result)

    def call_tool(
            self,
            name: str,
            arguments: Dict[str, Any]
    ) -> Union[CallToolResult, Generator[CallToolResult, None, Optional[CallToolResult]]]:
        params = CallToolRequestParams(
            name=name,
            arguments=arguments
        ).model_dump()
        result = self.call("tools/call", params)
        if not isinstance(result, GeneratorType):
            return CallToolResult.model_validate(result)
        else:
            def gen() -> Generator[CallToolResult, None, Optional[CallToolResult]]:
                it = iter(result)
                try:
                    while True:
                        item = next(it)
                        yield CallToolResult.model_validate(item)
                except StopIteration as e:
                    item = e.value
                    if item is not None:
                        return CallToolResult.model_validate(item)

            return gen()

    def list_resources(self) -> ListResourcesResult:
        result = self.call("resources/list")
        return ListResourcesResult.model_validate(result)

    def read_resource(
            self,
            uri: str
    ) -> Union[ReadResourceResult, Generator[ReadResourceResult, None, Optional[ReadResourceResult]]]:
        params = ReadResourceRequestParams(uri=uri).model_dump()
        result = self.call("resources/read", params)
        if not isinstance(result, GeneratorType):
            return ReadResourceResult.model_validate(result)
        else:
            def gen() -> Generator[ReadResourceResult, None, Optional[ReadResourceResult]]:
                it = iter(result)
                try:
                    while True:
                        item = next(it)
                        yield ReadResourceResult.model_validate(item)
                except StopIteration as e:
                    item = e.value
                    if item is not None:
                        return ReadResourceResult.model_validate(item)

            return gen()


class APIClient(SubroutineMixIn, MCPMixIn):

    def __init__(
            self,
            base_url: Optional[str] = None,
            *,
            headers: Optional[Dict[str, str]] = None,
            content_type: str = MIME.json.value,
            accept: str = f"{MIME.plain.value},{MIME.json.value},{MIME.sse.value}",
            user_agent: str = "python-libentry",
            connection: str = "keep-alive",
            api_key: Optional[str] = None,
            verify=False,
            stream_read_size: int = 512,
            sse_endpoint: str = "/sse",
            jsonrpc_endpoint: str = "/message"
    ) -> None:
        self.base_url = base_url

        self.headers = {} if headers is None else {**headers}
        self.headers["Content-Type"] = content_type
        self.headers["Accept"] = accept
        self.headers["User-Agent"] = user_agent
        self.headers["Connection"] = connection

        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.verify = verify
        self.stream_read_size = stream_read_size
        self.sse_endpoint = sse_endpoint
        self.jsonrpc_endpoint = jsonrpc_endpoint

        self.client = httpx.Client(verify=verify)

    @staticmethod
    def x_www_form_urlencoded(json_data: Dict[str, Any]):
        result = []
        for k, v in json_data.items():
            if v is not None:
                result.append((
                    k.encode("utf-8"),
                    v.encode("utf-8") if isinstance(v, str) else v,
                ))
        return urlencode(result, doseq=True)

    @staticmethod
    def find_content_type(*headers: Optional[Dict[str, str]]) -> Tuple[Optional[str], Dict[str, str]]:
        content_type = None
        for h in headers:
            if h is None:
                continue
            try:
                content_type = h["Content-Type"]
            except KeyError:
                continue

        if content_type is None:
            return None, {}

        items = content_type.split(";")
        mime = items[0].strip()
        params = {}
        for item in items[1:]:
            item = item.strip()
            i = item.find("=")
            if i < 0:
                continue
            params[item[:i]] = item[i + 1:]
        return mime, params

    def http_request(self, request: HTTPRequest) -> HTTPResponse:
        options = request.options
        timeout_value = options.timeout
        err = None
        for i in range(options.num_trials):
            timeout_value *= (1 + i * options.retry_factor)
            try:
                return self._http_request(request, timeout_value)
            except httpx.TimeoutException as e:
                err = e
                if callable(options.on_error):
                    options.on_error(e)
            except httpx.HTTPError as e:
                err = e
                if callable(options.on_error):
                    options.on_error(e)
            sleep(options.interval)
        raise err

    def _http_request(self, request: HTTPRequest, timeout: float) -> HTTPResponse:
        full_url = self.base_url.rstrip("/") + "/" + request.path.lstrip("/")
        headers = (
            {**self.headers}
            if request.options.headers is None else
            {**self.headers, **request.options.headers}
        )
        req_mime, _ = self.find_content_type(headers)
        if (req_mime is None) or req_mime in {MIME.json.value, MIME.plain.value}:
            payload = json.dumps(request.json_obj) if request.json_obj is not None else None
        elif req_mime == MIME.form.value:
            payload = self.x_www_form_urlencoded(request.json_obj) if request.json_obj is not None else None
        else:
            raise ValueError(f"Unsupported request MIME: \"{req_mime}\".")

        httpx_request = self.client.build_request(
            method=request.options.method,
            url=full_url,
            content=payload,
            headers=headers,
            timeout=timeout
        )
        httpx_response = self.client.send(httpx_request, stream=True)

        if httpx_response.status_code // 100 != 2:
            raise ServiceError(self._read_content(httpx_response))

        resp_mime, _ = self.find_content_type(httpx_response.headers)

        stream = request.options.stream
        if stream is None:
            stream = "-stream" in resp_mime

        if not stream:
            if resp_mime is None or resp_mime == MIME.plain.value:
                content = self._read_content(httpx_response)
            elif resp_mime == MIME.json.value:
                content = self._read_content(httpx_response)
                content = json.loads(content) if content else None
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")
        else:
            if resp_mime is None or resp_mime == MIME.sse.value:
                content = self._iter_events(self._iter_lines(httpx_response))
            elif resp_mime == MIME.json_stream.value:
                content = self._iter_objs(self._iter_lines(httpx_response))
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")
        return HTTPResponse(
            status_code=httpx_response.status_code,
            headers={**httpx_response.headers},
            stream=stream,
            content=content
        )

    # noinspection PyTypeChecker
    @staticmethod
    def _read_content(response: httpx.Response) -> str:
        try:
            charset = response.charset_encoding or "utf-8"
            return response.read().decode(charset)
        finally:
            response.close()

    @staticmethod
    def _iter_lines(response: httpx.Response) -> Generator[str, None, None]:
        try:
            for line in response.iter_lines():
                yield line
        finally:
            response.close()

    @staticmethod
    def _iter_events(lines: Iterable[str]) -> Generator[SSE, None, None]:
        decoder = SSEDecoder()
        for line in lines:
            line = line.rstrip()
            sse = decoder.decode(line)
            if sse is not None:
                yield sse

    @staticmethod
    def _iter_objs(lines: Iterable[str]) -> Generator[Dict, None, None]:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

    def subroutine_request(
            self,
            path: Union[str, Type[HasRequestPath], HasRequestPath, Any],
            params: Optional[Union[JSONObject, BaseModel]] = None,
            options: Optional[HTTPOptions] = None
    ) -> Union[SubroutineResponse, Generator[SubroutineResponse, None, None]]:
        if isinstance(path, BaseModel):
            if params is None:
                params = path
        if isinstance(params, BaseModel):
            params = params.model_dump(exclude_none=True)

        if hasattr(path, "get_request_path"):
            path = path.get_request_path()

        json_request = HTTPRequest(
            path=path,
            json_obj=params,
            options=options or HTTPOptions()
        )
        json_response = self.http_request(json_request)
        if not json_response.stream:
            return SubroutineResponse.model_validate(json_response.content)
        else:
            return self._iter_subroutine_responses(json_response)

    @staticmethod
    def _iter_subroutine_responses(response: HTTPResponse) -> Generator[SubroutineResponse, None, None]:
        return_response = None
        for sse in response.content:
            assert isinstance(sse, SSE)
            if sse.event == "message" and sse.data:
                json_obj = json.loads(sse.data)
                yield SubroutineResponse.model_validate(json_obj)
            elif sse.event == "return" and sse.data:
                json_obj = json.loads(sse.data)
                return_response = SubroutineResponse.model_validate(json_obj)
        return return_response

    def jsonrpc_request(
            self,
            request: JSONRPCRequest,
            path: Optional[str] = None,
            options: Optional[HTTPOptions] = None
    ) -> Union[JSONRPCResponse, Generator[JSONRPCResponse, None, None]]:
        json_request = HTTPRequest(
            path=path or self.jsonrpc_endpoint,
            json_obj=request.model_dump(),
            options=options or HTTPOptions()
        )
        json_response = self.http_request(json_request)
        if not json_response.stream:
            return JSONRPCResponse.model_validate(json_response.content)
        else:
            return self._iter_jsonrpc_responses(json_response)

    @staticmethod
    def _iter_jsonrpc_responses(response: HTTPResponse) -> Generator[JSONRPCResponse, None, None]:
        return_response = None
        for sse in response.content:
            assert isinstance(sse, SSE)
            if sse.event == "message" and sse.data:
                json_obj = json.loads(sse.data)
                yield JSONRPCResponse.model_validate(json_obj)
            elif sse.event == "return" and sse.data:
                json_obj = json.loads(sse.data)
                return_response = JSONRPCResponse.model_validate(json_obj)
        return return_response

    def jsonrpc_notify(
            self,
            request: JSONRPCNotification,
            path: Optional[str] = None,
            options: Optional[HTTPOptions] = None
    ) -> None:
        pass

    def start_session(self, sse_endpoint: Optional[str] = None):
        return SSESession(self, sse_endpoint=sse_endpoint or self.sse_endpoint)


class SSESession(MCPMixIn):

    def __init__(self, client: APIClient, sse_endpoint: str, sse_timeout: int = 6):
        self.client = client
        self.sse_endpoint = sse_endpoint
        self.sse_timeout = sse_timeout

        self.lock = Semaphore(0)
        self.endpoint = None
        self.pendings = {}
        self.closed = False

        self.ping_thread = Thread(target=self._ping_loop, daemon=False)
        self.ping_thread.start()

        self.sse_thread = Thread(target=self._sse_loop, daemon=False)
        self.sse_thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        with self.lock:
            self.closed = True

    def _ping_loop(self):
        interval = max(self.sse_timeout / 2, 0.5)
        while True:
            with self.lock:
                if self.closed:
                    break
            sleep(interval)
            self.ping()

    def _sse_loop(self):
        request = HTTPRequest(
            path=self.sse_endpoint,
            options=HTTPOptions(
                method="GET",
                timeout=self.sse_timeout
            )
        )
        response = self.client.http_request(request)
        assert response.stream
        type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCResponse, JSONRPCNotification])
        try:
            for sse in response.content:
                assert isinstance(sse, SSE)
                if sse.event == "endpoint":
                    self.endpoint = sse.data
                    self.lock.release()
                elif sse.event == "message":
                    json_obj = json.loads(sse.data)
                    obj = type_adapter.validate_python(json_obj)
                    if isinstance(obj, JSONRPCRequest):
                        self._on_request(obj)
                    elif isinstance(obj, JSONRPCNotification):
                        self._on_notification(obj)
                    elif isinstance(obj, JSONRPCResponse):
                        self._on_response(obj)
                    else:
                        pass
                else:
                    raise RuntimeError(f"Unknown event {sse.event}.")
                with self.lock:
                    if self.closed:
                        break
        except httpx.Timeout:
            pass

    def _on_request(self, request: JSONRPCRequest):
        print(request)
        pass

    def _on_notification(self, notification: JSONRPCNotification):
        print(notification)
        pass

    def _on_response(self, response: JSONRPCResponse):
        request_id = response.id
        with self.lock:
            pending = self.pendings.get(request_id)

        if pending is not None:
            pending.put(response)

    def jsonrpc_request(
            self,
            request: JSONRPCRequest,
            path: Optional[str] = None,
            options: Optional[HTTPOptions] = None
    ) -> JSONRPCResponse:
        with self.lock:
            if path is None:
                path = self.endpoint
            assert request.id not in self.pendings
            pending = Queue(8)
            self.pendings[request.id] = pending

        if options is None:
            options = HTTPOptions(stream=False)

        if options.stream is None or options.stream == True:
            raise ValueError(f"options.stream should be False.")

        self.client.http_request(HTTPRequest(
            path=path,
            json_obj=request.model_dump(),
            options=options
        ))

        response = pending.get()
        with self.lock:
            del self.pendings[request.id]

        if not isinstance(response, JSONRPCResponse):
            raise ServiceError(
                f"Invalid response type. "
                f"Expect JSONRPCResponse, got {type(response)}."
            )
        return response

    def jsonrpc_notify(
            self,
            request: JSONRPCNotification,
            path: Optional[str] = None,
            options: Optional[HTTPOptions] = None
    ) -> None:
        if path is None:
            with self.lock:
                path = self.endpoint

        if options is None:
            options = HTTPOptions(stream=False)

        if options.stream is None or options.stream == True:
            raise ValueError(f"options.stream should be False.")

        self.client.http_request(HTTPRequest(
            path=path,
            json_obj=request.model_dump(),
            options=options
        ))
