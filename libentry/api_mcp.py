#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "APIInfo",
    "route",
    "get",
    "post",
    "list_api_info",
    "ContentType",
    "JSONRPCRequest",
    "JSONRPCError",
    "JSONRPCResponse",
    "JSONRPCNotification",
    "SSEEvent",
    "APIClient",
]

import re
import traceback
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from queue import Queue
from threading import Semaphore, Thread
from time import sleep
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Tuple, Union
from urllib.parse import urlencode, urljoin

from pydantic import BaseModel, Field, TypeAdapter
from urllib3 import PoolManager
from urllib3.exceptions import HTTPError, TimeoutError

from libentry import json

API_INFO = "__api_info__"


@dataclass
class APIInfo:
    path: str = field()
    methods: List[str] = field()
    chunk_delimiter: str = field(default="\n\n")
    chunk_prefix: str = field(default=None)
    error_prefix: str = field(default="ERROR: ")
    extra_info: Mapping[str, Any] = field(default_factory=dict)


def route(
        path: Optional[str] = None,
        methods: List[str] = ("GET", "POST"),
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
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
            methods=methods,
            path=_path,
            chunk_delimiter=chunk_delimiter,
            chunk_prefix=chunk_prefix,
            extra_info=kwargs
        ))
        return fn

    return _api


get = partial(route, methods=["GET"])
post = partial(route, methods=["POST"])


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


class ServiceError(RuntimeError):

    def __init__(self, err: Union[str, Mapping[str, Any]]):
        try:
            err = json.loads(err)
        except ValueError:
            pass

        if isinstance(err, dict):
            if "message" in err:
                self.message = err.get("message")
                self.error = err.get("error")
                self.traceback = err.get("traceback")
            else:
                self.message = str(err)
                self.error = ""
                self.traceback = None
        else:
            self.message = err
            self.error = ""
            self.traceback = None

    def __str__(self):
        lines = []
        if self.message:
            lines += [self.message, "\n\n"]
        if self.error:
            lines += ["This is caused by server side error ", self.error, ".\n"]
        if self.traceback:
            lines += ["Below is the stacktrace:\n", self.traceback.rstrip()]
        return "".join(lines)


class ContentType(Enum):
    plain = "text/plain"
    form = "application/x-www-form-urlencoded"
    html = "text/html"
    # object type
    xml = "application/xml"
    json = "application/json"
    # stream type
    octet_stream = "application/octet-stream"
    json_stream = " application/json-stream"
    sse = "text/event-stream"


class HTTPRequest(BaseModel):
    url: str
    method: str
    headers: Mapping[str, str] = {}
    payload: Optional[Union[bytes, str]] = None
    timeout: float = 15
    num_trials: int = 5
    interval: float = 1
    retry_factor: float = 0.5
    on_error: Optional[Callable[[Exception], None]] = None
    stream: Optional[bool] = None
    verify: Optional[bool] = None


class HTTPResponse(BaseModel):
    status: int
    headers: Mapping[str, str] = {}
    content: Optional[Any]


class HTTPClient:

    def __init__(
            self,
            base_url: Optional[str] = None,
            headers: Optional[Mapping[str, str]] = None,
            content_type: str = ContentType.plain.value,
            accept: str = ContentType.plain.value,
            user_agent: str = "libentry.api_mcp.APIClient",
            connection: str = "close",
            api_key: Optional[str] = None,
            verify=False,
            stream_read_size: int = 512
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

    DEFAULT_CONN_POOL_SIZE = 10
    URLLIB3_POOL = (
        PoolManager(DEFAULT_CONN_POOL_SIZE, cert_reqs='CERT_NONE'),
        PoolManager(DEFAULT_CONN_POOL_SIZE)
    )

    @classmethod
    def reset_connection_pool(cls, size: int = 10):
        cls.URLLIB3_POOL = (
            PoolManager(size, cert_reqs='CERT_NONE'),
            PoolManager(size)
        )

    def request(self, request: HTTPRequest) -> HTTPResponse:
        err = None
        for i in range(request.num_trials):
            timeout = request.timeout * (1 + i * request.retry_factor)
            try:
                return self._single_request(request, timeout=timeout)
            except TimeoutError as e:
                err = e
                if callable(request.on_error):
                    request.on_error(e)
            except HTTPError as e:
                err = e
                if callable(request.on_error):
                    request.on_error(e)
            sleep(request.interval)
        raise err

    def _single_request(self, request: HTTPRequest, timeout: float) -> HTTPResponse:
        headers = {**self.headers, **request.headers}
        verify = self.verify if request.verify is None else request.verify
        client: PoolManager = self.URLLIB3_POOL[int(verify)]
        response = client.request(
            method=request.method,
            url=request.url,
            body=request.payload,
            headers=headers,
            timeout=timeout,
            preload_content=False
        )

        if response.status // 100 != 2:
            _, params = self.find_content_type(headers)
            charset = params.get("charset", "utf-8")
            raise ServiceError(self._load_bytes(response).decode(charset))

        headers = {**response.headers}
        content_type = headers.get("Content-Type", ContentType.plain.value)

        need_stream = "stream" in content_type or response.chunked
        stream = request.stream if request.stream is not None else need_stream

        if not stream:
            return HTTPResponse(
                status=response.status,
                headers=headers,
                content=self._load_bytes(response)
            )
        else:
            return HTTPResponse(
                status=response.status,
                headers=headers,
                content=self._iter_bytes(response, self.stream_read_size)
            )

    @staticmethod
    def _load_bytes(response) -> bytes:
        try:
            return response.data
        finally:
            response.release_conn()

    @staticmethod
    def _iter_bytes(response, read_size) -> Iterable[bytes]:
        try:
            if hasattr(response, "stream"):
                yield from response.stream(read_size)
            else:
                while True:
                    data = response.read(read_size)
                    if not data:
                        break
                    yield data
        finally:
            response.release_conn()

    @staticmethod
    def find_content_type(*headers: Optional[Mapping[str, str]]) -> Tuple[Optional[str], Mapping[str, str]]:
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


class JSONRPCRequest(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: Union[str, int] = Field()
    method: str = Field()
    params: Optional[Dict[str, Any]] = Field(default=None)


class JSONRPCError(BaseModel):
    code: int = Field(default=0)
    message: str = Field()
    data: Optional[Any] = Field(default=None)

    @classmethod
    def from_exception(cls, e):
        err_cls = e.__class__
        err_name = err_cls.__name__
        module = err_cls.__module__
        if module != "builtins":
            err_name = f"{module}.{err_name}"
        return cls(
            code=0,
            message=str(e),
            data={
                "error": err_name,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )


class JSONRPCResponse(BaseModel):
    jsonrpc: str = Field(default="2.0")
    id: Union[str, int] = Field()
    result: Optional[Any] = Field(default=None)
    error: Optional[JSONRPCError] = Field(default=None)


class JSONRPCNotification(BaseModel):
    jsonrpc: str = Field(default="2.0")
    method: str = Field()
    params: Optional[Dict[str, Any]] = Field(default=None)


class SSEEvent(BaseModel):
    event: str = Field()
    data: Optional[Any] = Field(default=None)


class SSEDecoder:
    def __init__(self) -> None:
        self._event = ""
        self._data: List[str] = []
        self._last_event_id = ""
        self._retry: Optional[int] = None

    def decode(self, line: str) -> Optional[SSEEvent]:
        if not line:
            if (
                    not self._event
                    and not self._data
                    and not self._last_event_id
                    and self._retry is None
            ):
                return None

            sse = SSEEvent(
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


class APIClient:

    def __init__(
            self,
            base_url: Optional[str] = None,
            headers: Optional[Mapping[str, str]] = None,
            content_type: str = ContentType.json.value,
            accept: str = f"{ContentType.plain.value},{ContentType.json.value},{ContentType.sse.value}",
            user_agent: str = "libentry.api_mcp.APIClient",
            connection: str = "close",
            api_key: Optional[str] = None,
            verify=False,
            stream_read_size: int = 512
    ) -> None:
        self.http_client = HTTPClient(
            base_url=base_url,
            headers=headers,
            content_type=content_type,
            accept=accept,
            user_agent=user_agent,
            connection=connection,
            api_key=api_key,
            verify=verify,
            stream_read_size=stream_read_size,
        )

    @staticmethod
    def x_www_form_urlencoded(json_data: Mapping[str, Any]):
        result = []
        for k, v in json_data.items():
            if v is not None:
                result.append((
                    k.encode("utf-8") if isinstance(k, str) else k,
                    v.encode("utf-8") if isinstance(v, str) else v,
                ))
        return urlencode(result, doseq=True)

    def request(
            self,
            method: Literal["GET", "POST"],
            path: str,
            json_data: Optional[Mapping] = None,
            *,
            headers: Optional[Mapping[str, str]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[Callable[[Exception], None]] = None,
            stream: Optional[bool] = None,
            stream_delimiter: Optional[str] = None,
            verify: Optional[bool] = None,
    ):
        full_url = urljoin(self.http_client.base_url, path)
        if headers is None:
            headers = {}
        req_mime, _ = self.http_client.find_content_type(headers, self.http_client.headers)
        if (req_mime is None) or req_mime in {ContentType.json.value, ContentType.plain.value}:
            payload = json.dumps(json_data) if json_data is not None else None
        elif req_mime == ContentType.form.value:
            payload = self.x_www_form_urlencoded(json_data) if json_data is not None else None
        else:
            raise ValueError(f"Unsupported request MIME: \"{req_mime}\".")

        response = self.http_client.request(HTTPRequest(
            url=full_url,
            method=method,
            headers=headers,
            payload=payload,
            timeout=timeout,
            interval=interval,
            num_trials=num_trials,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream,
            verify=verify
        ))

        content = response.content
        resp_mime, resp_params = self.http_client.find_content_type(response.headers)
        resp_charset = resp_params.get("charset", "utf-8")
        if not isinstance(content, GeneratorType):
            if not isinstance(content, bytes):
                raise ServiceError(
                    f"Content type missmatch, "
                    f"expected bytes, got {type(content)}."
                )
            content = content.decode(resp_charset)
            if resp_mime is None or resp_mime == ContentType.plain.value:
                return content
            elif resp_mime == ContentType.json.value:
                try:
                    return json.loads(content)
                except ValueError:
                    # fallback to raw text
                    return content
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")
        else:
            if resp_mime is None or resp_mime == ContentType.sse.value:
                if stream_delimiter is None:
                    stream_delimiter = "\n"
                chunks = self._iter_chunks(content, stream_delimiter)
                return self._iter_events(chunks, resp_charset)
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")

    @staticmethod
    def _iter_chunks(bytes_stream: Iterable[bytes], delimiter: Union[bytes, str]) -> Iterable[bytes]:
        if isinstance(delimiter, str):
            delimiter = delimiter.encode()

        pending: Optional[bytes] = None
        for data in bytes_stream:
            if pending is not None:
                data = pending + data

            chunks = data.split(delimiter)
            pending = chunks.pop() if chunks and chunks[-1] and chunks[-1][-1] == data[-1] else None

            for chunk in chunks:
                yield chunk

        if pending is not None:
            yield pending

    @staticmethod
    def _iter_events(chunks: Iterable[bytes], charset: str) -> Iterable[SSEEvent]:
        decoder = SSEDecoder()
        for line in chunks:
            line = line.rstrip().decode(charset)
            sse = decoder.decode(line)
            if sse is not None:
                yield sse

        # delimiter = "\n"
        # evnet_prefix = "event:"
        # data_prefix = "data:"
        # for chunk in chunks:
        #     chunk = chunk.strip()
        #     if not chunk:
        #         continue
        #
        #     lines = chunk.decode(charset).split(delimiter)
        #     if len(lines) == 2:
        #         event_line, data_line = lines
        #         if not (event_line.startswith(evnet_prefix) and data_line.startswith(data_prefix)):
        #             raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")
        #         event = event_line[len(evnet_prefix):]
        #         data = data_line[len(data_prefix):]
        #     elif len(lines) == 1:
        #         line = lines[0]
        #         if line.startswith(evnet_prefix):
        #             event = line[len(evnet_prefix):]
        #             data = None
        #         elif line.startswith(data_prefix):
        #             event = "message"
        #             data = line[len(data_prefix):]
        #         else:
        #             event = "message"
        #             data = line
        #     else:
        #         raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")
        #
        #     yield SSEEvent(event=event, data=data)

    def call(
            self,
            name: str,
            params: Optional[Mapping[str, Any]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[Callable[[Exception], None]] = None,
            stream: Optional[bool] = None
    ) -> Any:
        request_id = str(uuid.uuid4())
        request: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": name,
        }
        if params is not None:
            request["params"] = params
        response = self.request(
            "POST",
            "/" + name,
            json_data=request,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream
        )
        if not isinstance(response, GeneratorType):
            if not isinstance(response, Mapping):
                raise ServiceError(
                    f"Invalid response type. "
                    f"Expected JSON object, got {type(response)}."
                )
            return self._get_result(response)
        else:
            return self._iter_results(response)

    @staticmethod
    def _get_result(response: Mapping[str, Any]):
        resp = JSONRPCResponse.model_validate(response)
        if resp.error is not None:
            raise ServiceError(resp.error.data)
        if resp.result is not None:
            return resp.result
        else:
            raise ServiceError("The response doesn't contain any result.")

    @staticmethod
    def _iter_results(response: Iterable[SSEEvent]):
        for sse in response:
            if not isinstance(sse, SSEEvent):
                raise ServiceError("Not a valid event stream.")

            if sse.event != "message" or sse.data is None:
                continue

            resp = JSONRPCResponse.model_validate(json.loads(sse.data))
            if resp.error is not None:
                raise ServiceError(resp.error.data)
            if resp.result is not None:
                yield resp.result
            else:
                raise ServiceError("The response doesn't contain any result.")

    def start_session(self):
        return SSESession(self)


class SSESession:

    def __init__(self, client: APIClient):
        self.client = client

        self.sse_thread = Thread(target=self._sse_loop, daemon=True)
        self.sse_thread.start()

        self.lock = Semaphore(0)
        self.endpoint = None
        self.pendings = {}

    def _sse_loop(self):
        response = self.client.request("GET", "/sse", timeout=60, stream=True)
        type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCResponse, JSONRPCNotification])
        for sse in response:
            if sse.event == "endpoint":
                self.endpoint = urljoin(self.client.http_client.base_url, sse.data)
                self.lock.release()
            elif sse.event == "message":
                obj = type_adapter.validate_python(json.loads(sse.data))
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

    def _on_request(self, request: JSONRPCRequest):
        pass

    def _on_notification(self, notification: JSONRPCNotification):
        pass

    def _on_response(self, response: JSONRPCResponse):
        request_id = response.id
        with self.lock:
            pending = self.pendings.get(request_id)

        if pending is not None:
            pending.put(response)

    def call(
            self,
            name: str,
            params: Optional[Mapping[str, Any]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Any:
        request_id = str(uuid.uuid4())
        with self.lock:
            endpoint = self.endpoint
            assert request_id not in self.pendings
            pending = Queue(8)
            self.pendings[request_id] = pending

        request = JSONRPCRequest(id=request_id, method=name, params=params)
        self.client.request(
            "POST",
            endpoint,
            json_data=request.model_dump(exclude_none=True),
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=False
        )

        response = pending.get()
        with self.lock:
            del self.pendings[request_id]

        if not isinstance(response, JSONRPCResponse):
            raise ServiceError(
                f"Invalid response type. "
                f"Expect JSONRPCResponse, got {type(response)}."
            )

        if response.error is not None:
            raise ServiceError(response.error.data)
        if response.result is not None:
            return response.result
        else:
            raise ServiceError("The response doesn't contain any result.")

    def notify(
            self,
            name: str,
            params: Optional[Mapping[str, Any]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[Callable[[Exception], None]] = None,
    ) -> Any:
        with self.lock:
            endpoint = self.endpoint

        request = JSONRPCNotification(method=name, params=params)
        self.client.request(
            "POST",
            endpoint,
            json_data=request.model_dump(exclude_none=True),
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=False
        )


    def initialize(self):
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "libentry-client", "version": "1.0.0"}
        }
        result = self.call("initialize", params)
        print(result)
        self.notify("notifications/initialized")
