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
from typing import Any, AsyncIterable, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Tuple, Union
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
    xml = "application/xml"
    json = "application/json"
    sse = "text/event-stream"


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


class SSEStream:

    def __init__(self, delimiter, charset):
        self.delimiter = delimiter.encode(charset) if delimiter is not None else None
        self.charset = charset

    def iter_chunks(self, data_list: Iterable[bytes]) -> Iterable[str]:
        pending = None
        for data in data_list:
            if pending is not None:
                data = pending + data

            chunks = data.split(self.delimiter) if self.delimiter else data.splitlines()
            pending = chunks.pop() if chunks and chunks[-1] and chunks[-1][-1] == data[-1] else None

            for chunk in chunks:
                yield chunk.decode(self.charset)

        if pending is not None:
            yield pending.decode(self.charset)

    def iter_events(self, chunks: Iterable[str]):
        error = None
        for chunk in chunks:
            if error is not None:
                # error is not None means there is a fatal exception raised from the server side.
                # The client should just complete the stream and then raise the error to the upper.
                continue

            if not chunk:
                continue

            decoded = self._decode_chunk(chunk)
            if decoded["event"] == "message":
                try:
                    decoded["data"] = json.loads(decoded["data"])
                except ValueError:
                    pass
            yield decoded

        if error is not None:
            raise error

    def _decode_chunk(self, chunk: str) -> Dict[str, str]:
        lines = chunk.strip().split("\n")
        if len(lines) == 2:
            event_line, data_line = lines
            if not (event_line.startswith("event:") and data_line.startswith("data:")):
                raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")
            return {"event": event_line[len("event:"):], "data": data_line[len("data:"):]}
        elif len(lines) == 1:
            line = lines[0]
            if line.startswith("event:"):
                return {"event": line[len("event:"):]}
            elif line.startswith("data:"):
                return {"event": "message", "data": line[len("data:"):]}
            else:
                return {"event": "message", "data": line}
        else:
            raise RuntimeError(f"Failed to decode chunk: \"{chunk}\".")


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
    data: Any = Field()


class APIClient(BaseClient):

    @staticmethod
    def _encode_params(data):
        result = []
        for k, v in data.items():
            if v is not None:
                result.append(
                    (
                        k.encode("utf-8") if isinstance(k, str) else k,
                        v.encode("utf-8") if isinstance(v, str) else v,
                    )
                )
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
            chunk_delimiter: str = "\n\n",
            verify: Optional[bool] = None,
    ):
        full_url = urljoin(self.base_url, path)
        headers = {**headers} if headers else {}
        content_type = headers.get("Content-Type", self.headers.get("Content-Type"))
        if content_type is None or content_type == ContentType.json.value:
            body = json.dumps(json_data) if json_data is not None else None
        elif content_type == ContentType.form.value:
            body = self._encode_params(json_data)
            headers["Content-Type"] = ContentType.json.value
        else:
            raise ValueError(f"Unsupported content type \"{content_type}\".")

        content = super().request(
            method,
            full_url,
            body=body,
            headers=headers,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream,
            verify=verify,
        )

        if not isinstance(content, GeneratorType):
            return _load_json_or_str(content.decode(self.charset))
        else:
            sse = SSEStream(chunk_delimiter, self.charset)
            return sse.iter_events(sse.iter_chunks(content))

    def get(
            self,
            path: Optional[str] = None,
            *,
            headers: Optional[Mapping[str, str]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            stream: bool = False,
            chunk_delimiter: str = "\n\n",
            chunk_prefix: str = None,
            chunk_suffix: str = None,
            error_prefix: str = "ERROR: ",
    ):
        return self.request(
            "GET",
            path,
            headers=headers,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream,
            chunk_delimiter=chunk_delimiter,
            chunk_prefix=chunk_prefix,
            chunk_suffix=chunk_suffix,
            error_prefix=error_prefix,
        )

    def post(
            self,
            path: Optional[str] = None,
            json_data: Optional[Mapping] = None,
            *,
            headers: Optional[Mapping[str, str]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            stream: bool = False,
            chunk_delimiter: str = "\n\n",
            chunk_prefix: str = None,
            chunk_suffix: str = None,
            error_prefix: str = "ERROR: ",
    ):
        return self.request(
            "POST",
            path,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream,
            chunk_delimiter=chunk_delimiter,
            chunk_prefix=chunk_prefix,
            chunk_suffix=chunk_suffix,
            error_prefix=error_prefix,
        )

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
            def _stream():
                for item in response:
                    event = SSEEvent.model_validate(item)
                    if event.event != "message" or event.data is None:
                        continue

                    resp = JSONRPCResponse.model_validate(event.data)
                    if resp.error is not None:
                        raise ServiceError(resp.error.data)
                    if resp.result is not None:
                        yield resp.result
                    else:
                        raise ServiceError("The response doesn't contain any result.")

            return _stream()

    async def request_async(
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
            stream: bool = False,
            chunk_delimiter: str = "\n\n",
            chunk_prefix: str = None,
            chunk_suffix: str = None,
            error_prefix: str = "ERROR: ",
            exhaust_stream: bool = False,
            verify: Optional[bool] = None,
    ):
        full_url = urljoin(self.base_url, path)
        headers = {**headers} if headers else {}
        content_type = headers.get("Content-Type", self.headers.get("Content-Type"))
        if content_type is None or content_type == "application/json":
            body = json.dumps(json_data) if json_data is not None else None
        elif content_type == "application/x-www-form-urlencoded":
            body = self._encode_params(json_data)
        else:
            raise ValueError(f"Unsupported content type \"{content_type}\".")

        content = await super().request_async(
            method,
            full_url,
            body=body,
            headers=headers,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream,
            verify=verify,
        )
        if not stream:
            assert isinstance(content, bytes)
            return _load_json_or_str(content.decode(self.charset))
        else:
            async def iter_lines(data_list: AsyncIterable[bytes]) -> AsyncIterable[str]:
                delimiter = chunk_delimiter.encode(self.charset) if chunk_delimiter is not None else None
                pending = None
                async for data in data_list:
                    if pending is not None:
                        data = pending + data

                    lines = data.split(delimiter) if delimiter else data.splitlines()
                    pending = lines.pop() if lines and lines[-1] and lines[-1][-1] == data[-1] else None

                    for line in lines:
                        yield line.decode(self.charset)

                if pending is not None:
                    yield pending.decode(self.charset)

            async def iter_chunks(lines: AsyncIterable[str]) -> AsyncIterable:
                error = None
                async for chunk in lines:
                    if error is not None:
                        # error is not None means there is a fatal exception raised from the server side.
                        # The client should just complete the stream and then raise the error to the upper.
                        continue

                    if not chunk:
                        continue

                    if error_prefix is not None:
                        if chunk.startswith(error_prefix):
                            chunk = chunk[len(error_prefix):]
                            error = ServiceError(chunk)
                            continue

                    if chunk_prefix is not None:
                        if chunk.startswith(chunk_prefix):
                            chunk = chunk[len(chunk_prefix):]
                        else:
                            continue

                    if chunk_suffix is not None:
                        if chunk.endswith(chunk_suffix):
                            chunk = chunk[:-len(chunk_suffix)]
                        else:
                            continue

                    yield _load_json_or_str(chunk)

                if error is not None:
                    raise error

            gen = iter_lines(content)
            gen = iter_chunks(gen)
            return gen if not exhaust_stream else [*gen]

    async def get_async(
            self,
            path: Optional[str] = None,
            *,
            headers: Optional[Mapping[str, str]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None
    ):
        return await self.request_async(
            "GET",
            path,
            headers=headers,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
        )

    async def post_async(
            self,
            path: Optional[str] = None,
            json_data: Optional[Mapping] = None,
            *,
            headers: Optional[Mapping[str, str]] = None,
            timeout: float = 15,
            num_trials: int = 5,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            stream: bool = False,
            chunk_delimiter: str = "\n\n",
            chunk_prefix: str = None,
            chunk_suffix: str = None,
            error_prefix: str = "ERROR: ",
            exhaust_stream: bool = False,
    ):
        return await self.request_async(
            "POST",
            path,
            json_data=json_data,
            headers=headers,
            timeout=timeout,
            num_trials=num_trials,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error,
            stream=stream,
            chunk_delimiter=chunk_delimiter,
            chunk_prefix=chunk_prefix,
            chunk_suffix=chunk_suffix,
            error_prefix=error_prefix,
            exhaust_stream=exhaust_stream,
        )

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
