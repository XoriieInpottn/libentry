#!/usr/bin/env python3

__author__ = "xi"

import uuid
from queue import Queue
from threading import Semaphore, Thread
from time import sleep
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlencode, urljoin

import httpx
from pydantic import TypeAdapter

from libentry import json
from libentry.mcp.types import JSONRPCNotification, JSONRPCRequest, JSONRPCResponse, JSONRequest, JSONResponse, MIME, \
    SSE, ServiceError, _JSONRequest


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


class APIClient:

    def __init__(
            self,
            base_url: Optional[str] = None,
            headers: Optional[Dict[str, str]] = None,
            content_type: str = MIME.json.value,
            accept: str = f"{MIME.plain.value},{MIME.json.value},{MIME.sse.value}",
            user_agent: str = "python-libentry",
            connection: str = "keep-alive",
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

        self.client = httpx.Client(verify=verify)

    @staticmethod
    def x_www_form_urlencoded(json_data: Dict[str, Any]):
        result = []
        for k, v in json_data.items():
            if v is not None:
                result.append((
                    k.encode("utf-8") if isinstance(k, str) else k,
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

    def request(self, request: JSONRequest) -> JSONResponse:
        single_request = request.model_copy()
        err = None
        for i in range(request.num_trials):
            single_request.timeout *= (1 + i * request.retry_factor)
            try:
                return self._request(single_request)
            except httpx.TimeoutException as e:
                err = e
                if callable(request.on_error):
                    request.on_error(e)
            except httpx.HTTPError as e:
                err = e
                if callable(request.on_error):
                    request.on_error(e)
            sleep(request.interval)
        raise err

    def _request(self, request: _JSONRequest) -> JSONResponse:
        full_url = urljoin(self.base_url, request.path)
        headers = (
            {**self.headers}
            if request.headers is None else
            {**self.headers, **request.headers}
        )
        req_mime, _ = self.find_content_type(headers)
        if (req_mime is None) or req_mime in {MIME.json.value, MIME.plain.value}:
            payload = json.dumps(request.json_obj) if request.json_obj is not None else None
        elif req_mime == MIME.form.value:
            payload = self.x_www_form_urlencoded(request.json_obj) if request.json_obj is not None else None
        else:
            raise ValueError(f"Unsupported request MIME: \"{req_mime}\".")

        httpx_request = self.client.build_request(
            method=request.method,
            url=full_url,
            content=payload,
            headers=headers,
            timeout=request.timeout
        )
        httpx_response = self.client.send(httpx_request, stream=True)

        if httpx_response.status_code // 100 != 2:
            raise ServiceError(self._read_content(httpx_response))

        resp_mime, _ = self.find_content_type(httpx_response.headers)

        stream = request.stream
        if stream is None:
            stream = "-stream" in resp_mime

        if not stream:
            if resp_mime is None or resp_mime == MIME.plain.value:
                content = self._read_content(httpx_response)
            elif resp_mime == MIME.json.value:
                content = json.loads(self._read_content(httpx_response))
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")
        else:
            if resp_mime is None or resp_mime == MIME.sse.value:
                content = self._iter_events(self._iter_lines(httpx_response))
            elif resp_mime == MIME.json_stream.value:
                content = self._iter_objs(self._iter_lines(httpx_response))
            else:
                raise RuntimeError(f"Unsupported response MIME: \"{resp_mime}\".")
        return JSONResponse(
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
    def _iter_lines(response: httpx.Response) -> Iterable[str]:
        try:
            for line in response.iter_lines():
                yield line
        finally:
            response.close()

    @staticmethod
    def _iter_events(lines: Iterable[str]) -> Iterable[SSE]:
        decoder = SSEDecoder()
        for line in lines:
            line = line.rstrip()
            sse = decoder.decode(line)
            if sse is not None:
                yield sse

    @staticmethod
    def _iter_objs(lines: Iterable[str]) -> Iterable[Dict]:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

    def rpc_request(
            self,
            request: JSONRPCRequest,
            path: Optional[str] = None
    ) -> Union[JSONRPCResponse, Iterable[JSONRPCResponse]]:
        json_request = JSONRequest(
            method="POST",
            path=path or f"/{request.method}",
            json_obj=request.model_dump(),
        )
        json_response = self.request(json_request)
        if not json_response.stream:
            return JSONRPCResponse.model_validate(json_response.content)
        else:
            return self._iter_rpc_response(json_response)

    @staticmethod
    def _iter_rpc_response(response: JSONResponse) -> Iterable[JSONRPCResponse]:
        for sse in response.content:
            assert isinstance(sse, SSE)
            if sse.event != "message":
                continue
            if not sse.data:
                continue
            json_obj = json.loads(sse.data)
            yield JSONRPCResponse.model_validate(json_obj)


class SSESession:

    def __init__(self, client: APIClient, sse_endpoint: str = "/sse"):
        self.client = client
        self.sse_endpoint = sse_endpoint

        self.sse_thread = Thread(target=self._sse_loop, daemon=True)
        self.sse_thread.start()

        self.lock = Semaphore(0)
        self.endpoint = None
        self.pendings = {}

    def _sse_loop(self):
        request = JSONRequest(
            method="GET",
            path=self.sse_endpoint,
            timeout=60,
        )
        response = self.client.request(request)
        assert response.stream
        type_adapter = TypeAdapter(Union[JSONRPCRequest, JSONRPCResponse, JSONRPCNotification])
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

    def rpc_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        with self.lock:
            endpoint = self.endpoint
            assert request.id not in self.pendings
            pending = Queue(8)
            self.pendings[request.id] = pending

        self.client.request(JSONRequest(
            method="POST",
            path=endpoint,
            json_obj=request.model_dump(),
            stream=False
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

    def rpc_notify(self, request: JSONRPCNotification):
        with self.lock:
            endpoint = self.endpoint

        self.client.request(JSONRequest(
            method="POST",
            path=endpoint,
            json_obj=request.model_dump(),
            stream=False
        ))

    def initialize(self):
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "libentry-client", "version": "1.0.0"}
        }
        response = self.rpc_request(JSONRPCRequest(method="initialize", id=str(uuid.uuid4()), params=params))
        print(response.result)
        self.rpc_notify(JSONRPCNotification(method="notifications/initialized"))
