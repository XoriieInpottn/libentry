#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "APIInfo",
    "route",
    "get",
    "post",
    "list_api_info",
    "ContentType",
    "APIClient",
]

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from threading import Lock, Thread
from time import sleep
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Tuple, Union
from urllib.parse import urlencode, urljoin

import httpx
from pydantic import BaseModel, Field
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


def _load_json_or_str(text: str) -> Union[Mapping[str, Any], str]:
    try:
        return json.loads(text)
    except ValueError:
        return text


class ServiceError(RuntimeError):

    def __init__(self, text: Union[str, Mapping[str, Any]]):
        err = text if isinstance(text, Mapping) else _load_json_or_str(text)
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


ErrorCallback = Callable[[Exception], None]


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
            content_type: str = f"{ContentType.plain}",
            accept: str = f"{ContentType.plain}",
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

        if response.status != 200:
            _, params = self.find_content_type(headers)
            charset = params.get("charset", "utf-8")
            try:
                raise ServiceError(response.data.decode(charset))
            finally:
                response.release_conn()

        headers = {**response.headers}
        content_type = headers.get("Content-Type", ContentType.plain.value)

        need_stream = "stream" in content_type or response.chunked
        stream = request.stream if request.stream is not None else need_stream

        if not stream:
            try:
                return HTTPResponse(
                    status=response.status,
                    headers=headers,
                    content=response.data
                )
            finally:
                response.release_conn()
        else:
            def iter_bytes() -> Iterable[bytes]:
                try:
                    if hasattr(response, "stream"):
                        yield from response.stream(self.stream_read_size)
                    else:
                        while True:
                            data = response.read(self.stream_read_size)
                            if not data:
                                break
                            yield data
                finally:
                    response.release_conn()

            return HTTPResponse(
                status=response.status,
                headers=headers,
                content=iter_bytes()
            )

    def find_content_type(self, headers: Optional[Mapping[str, str]]) -> Tuple[Optional[str], Mapping[str, str]]:
        if headers is not None and "Content-Type" in headers:
            content_type = headers["Content-Type"]
        else:
            content_type = self.headers.get("Content-Type")

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


class BaseClient:

    def __init__(
            self,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            content_type: str = ContentType.json.value,
            accept: str = f"{ContentType.plain},{ContentType.json},{ContentType.sse}",
            user_agent: str = "libentry.api_mcp.APIClient",
            connection: str = "close",
            headers: Optional[Mapping[str, str]] = None,
            verify=False,
            stream_read_size: int = 512,
            charset: str = "UTF-8",
    ) -> None:
        self.base_url = base_url
        self.headers = {
            "Accept": accept,
            "Content-Type": content_type,
            "User-Agent": user_agent,
            "Connection": connection,
        }

        if headers is not None:
            self.headers.update(headers)

        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"

        self.verify = verify
        self.stream_read_size = stream_read_size
        self.charset = charset

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

    def _single_request(
            self,
            method: str,
            url: str,
            body: Optional[Union[bytes, str]],
            headers: Optional[Mapping[str, str]],
            timeout: float,
            stream: Optional[bool],
            verify: bool,
    ):
        headers = self.headers if headers is None else {**self.headers, **headers}
        client: PoolManager = self.URLLIB3_POOL[int(verify)]
        response = client.request(
            method=method,
            url=url,
            body=body,
            headers=headers,
            timeout=timeout,
            preload_content=False
        )

        if response.status != 200:
            try:
                raise ServiceError(response.data.decode(self.charset))
            finally:
                response.release_conn()

        if stream is None:
            stream = "stream" in response.headers.get("Content-Type") or response.chunked

        if not stream:
            try:
                return response.data
            finally:
                response.release_conn()
        else:
            def iter_content():
                try:
                    if hasattr(response, "stream"):
                        yield from response.stream(self.stream_read_size, decode_content=True)
                    else:
                        while True:
                            data = response.read(self.stream_read_size)
                            if not data:
                                break
                            yield data
                finally:
                    response.release_conn()

            return iter_content()

    def request(
            self,
            method: Literal["GET", "POST"],
            url: str,
            body: Optional[Union[bytes, str]] = None,
            headers: Optional[Mapping[str, str]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            stream: Optional[bool] = None,
            verify: Optional[bool] = None,
    ) -> Union[bytes, Iterable[bytes]]:
        headers = self.headers if headers is None else headers
        verify = self.verify if verify is None else verify

        err = None
        for i in range(num_trials):
            try:
                return self._single_request(
                    method=method,
                    url=url,
                    body=body,
                    headers=headers,
                    timeout=timeout * (1 + i * retry_factor),
                    stream=stream,
                    verify=verify,
                )
            except TimeoutError as e:
                err = e
                if callable(on_error):
                    on_error(e)
            except HTTPError as e:
                err = e
                if callable(on_error):
                    on_error(e)
            sleep(interval)
        raise err

    HTTPX_POOL = httpx.Limits(max_keepalive_connections=DEFAULT_CONN_POOL_SIZE)

    async def _single_request_async(
            self,
            method: str,
            url: str,
            body: Optional[Union[bytes, str]],
            headers: Optional[Mapping[str, str]],
            timeout: float,
            stream: bool,
            verify: bool,
    ):
        headers = self.headers if headers is None else {**self.headers, **headers}
        if not stream:
            async with httpx.AsyncClient(headers=headers, verify=verify) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    content=body,
                    timeout=timeout
                )
                try:
                    if response.status_code != 200:
                        raise ServiceError(response.content.decode(self.charset))
                    return response.content
                finally:
                    await response.aclose()
        else:
            client = httpx.AsyncClient(headers=headers, verify=verify)
            response = client.stream(
                method=method,
                url=url,
                content=body,
                timeout=timeout
            )

            async def iter_content():
                try:
                    async with response as r:
                        if r.status_code != 200:
                            content = await r.aread()
                            raise ServiceError(content.decode(self.charset))
                        async for data in r.aiter_bytes():
                            yield data
                finally:
                    await client.aclose()

            return iter_content()

    async def request_async(
            self,
            method: Literal["GET", "POST"],
            url: str,
            body: Optional[Union[bytes, str]] = None,
            headers: Optional[Mapping[str, str]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            stream: bool = False,
            verify: Optional[bool] = None,
    ):
        headers = self.headers if headers is None else headers
        verify = self.verify if verify is None else verify

        err = None
        for i in range(num_trials):
            try:
                return await self._single_request_async(
                    method=method,
                    url=url,
                    body=body,
                    headers=headers,
                    timeout=timeout * (1 + i * retry_factor),
                    stream=stream,
                    verify=verify,
                )
            except httpx.TimeoutException as e:
                err = e
                if callable(on_error):
                    on_error(e)
            except httpx.HTTPError as e:
                err = e
                if callable(on_error):
                    on_error(e)
            await asyncio.sleep(interval)
        raise err


class JSONRPCError(BaseModel):
    code: int = Field(default=0)
    message: str = Field()
    data: Optional[Any] = Field(default=None)


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


class ChunkStream:

    def __init__(self, delimiter, charset):
        self.delimiter = delimiter.encode(charset) if delimiter is not None else None
        self.charset = charset

    def __call__(self, stream: Iterable[bytes]) -> Iterable[str]:
        pending: Optional[bytes] = None
        for data in stream:
            if pending is not None:
                data = pending + data

            chunks = data.split(self.delimiter) if self.delimiter else data.splitlines()
            pending = chunks.pop() if chunks and chunks[-1] and chunks[-1][-1] == data[-1] else None

            for chunk in chunks:
                yield chunk.decode(self.charset)

        if pending is not None:
            yield pending.decode(self.charset)


class SSEStream:

    def __init__(self, delimiter, charset):
        self.chunk_stream = ChunkStream(delimiter, charset)

    def __call__(self, stream: Iterable[bytes]):
        for chunk in self.chunk_stream(stream):
            chunk = chunk.strip()
            if not chunk:
                continue

            yield self._decode_chunk(chunk)

    def _decode_chunk(self, chunk: str) -> SSEEvent:
        lines = chunk.strip().split("\n")
        if len(lines) == 2:
            event_line, data_line = lines
            if not (event_line.startswith("event:") and data_line.startswith("data:")):
                raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")
            event = event_line[len("event:"):]
            data = data_line[len("data:"):]
        elif len(lines) == 1:
            line = lines[0]
            if line.startswith("event:"):
                event = line[len("event:"):]
                data = None
            elif line.startswith("data:"):
                event = "message"
                data = line[len("data:"):]
            else:
                event = "message"
                data = line
        else:
            raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")

        return SSEEvent(event=event, data=data)


class APIClient:

    def __init__(
            self,
            base_url: Optional[str] = None,
            headers: Optional[Mapping[str, str]] = None,
            content_type: str = ContentType.json.value,
            accept: str = f"{ContentType.json},{ContentType.sse}",
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
            on_error: Optional[ErrorCallback] = None,
            stream: Optional[bool] = None,
            stream_delimiter: Optional[str] = None,
            verify: Optional[bool] = None,
    ):
        full_url = urljoin(self.http_client.base_url, path)
        req_mime, _ = self.http_client.find_content_type(headers)
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
                return _load_json_or_str(content)
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")
        else:
            if resp_mime is None or resp_mime == ContentType.sse.value:
                if stream_delimiter is None:
                    stream_delimiter = "\n\n"
                chunks = self.iter_chunks(content, stream_delimiter)
                return self.iter_events(chunks, resp_charset)
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")

    @staticmethod
    def iter_chunks(bytes_stream: Iterable[bytes], delimiter: Union[bytes, str]) -> Iterable[bytes]:
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
    def iter_events(chunks: Iterable[bytes], charset: str) -> Iterable[SSEEvent]:
        d = "\n"
        ep = "event:"
        dp = "data:"
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            lines = chunk.decode(charset).split(d)
            if len(lines) == 2:
                event_line, data_line = lines
                if not (event_line.startswith(ep) and data_line.startswith(dp)):
                    raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")
                event = event_line[len(ep):]
                data = data_line[len(dp):]
            elif len(lines) == 1:
                line = lines[0]
                if line.startswith(ep):
                    event = line[len(ep):]
                    data = None
                elif line.startswith(dp):
                    event = "message"
                    data = line[len(dp):]
                else:
                    event = "message"
                    data = line
            else:
                raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")

            yield SSEEvent(event=event, data=data)

    def call(
            self,
            name: str,
            params: Optional[Mapping[str, Any]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            stream: Optional[bool] = None
    ) -> Any:
        request_id = str(uuid.uuid4())
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": name,
        }
        if params is not None:
            request["params"] = params
        accepts = [ContentType.plain.value, ContentType.json.value, ContentType.sse.value]
        response = self.request(
            "POST",
            "/" + name,
            headers={"Accept": ",".join(accepts)},
            json_data=request,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream
        )

        if not isinstance(response, GeneratorType):
            resp = JSONRPCResponse.model_validate(response)
            if resp.error is not None:
                raise ServiceError(resp.error.data)
            if resp.result is not None:
                return resp.result
            else:
                raise ServiceError("The response doesn't contain any result.")
        else:
            return self.iter_results(response)

    @staticmethod
    def iter_results(response: Iterable[SSEEvent]):
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

        self.lock = Lock()
        self.endpoint = None
        self.pendings = {}

    def _sse_loop(self):
        response = self.client.get(
            "/sse",
            headers={"Accept": ContentType.sse.value},
            timeout=60,
            stream=True
        )
        for chunk in response:
            event, data = self._parse_event(chunk)
            if event is None:
                continue
            if event == "endpoint":
                with self.lock:
                    self.endpoint = urljoin(self.client.base_url, data)
            elif event == "message":
                if isinstance(data, str):
                    response = json.loads(data)
                request_id = data
                with self.lock:
                    pass
            else:
                raise RuntimeError(f"Unknown event {event}.")
            print(event)
            print(data)

    def _parse_event(self, chunk):
        if isinstance(chunk, str):
            chunk = chunk.strip()
            i = chunk.find("event:")
            if i < 0:
                return None, None

            j = chunk.find("\ndata:")
            if j < 0:
                event = chunk[i + len("event:"):].strip()
                data = None
            else:
                event = chunk[i + len("event:"):j].strip()
                data = chunk[j + len("\ndata:"):]
            return event, data
        elif isinstance(chunk, dict):
            if "event" in chunk and "data" in chunk:
                return chunk["event"], chunk["data"]
        return None, None

    def initialize(self):
        pass
