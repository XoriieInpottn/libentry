#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "APIInfo",
    "api",
    "get",
    "post",
    "list_api_info",
    "APIClient",
]

from dataclasses import dataclass, field
from time import sleep
from typing import Any, Callable, Iterable, List, Literal, Mapping, Optional, Tuple
from urllib.parse import urljoin

from urllib3 import PoolManager
from urllib3.exceptions import HTTPError, TimeoutError

from libentry import json

API_INFO = "__api_info__"


@dataclass
class APIInfo:
    method: str = field()
    path: str = field()
    mime_type: str = field(default="application/json")
    chunk_delimiter: str = field(default="\n\n")
    chunk_prefix: str = field(default=None)
    chunk_suffix: str = field(default=None)
    stream_prefix: str = field(default=None)
    stream_suffix: str = field(default=None)
    error_prefix: str = field(default="ERROR: ")
    extra_info: Mapping[str, Any] = field(default_factory=dict)


def api(
        method: Literal["GET", "POST"] = "POST",
        path: Optional[str] = None,
        mime_type: str = "application/json",
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
        chunk_suffix: str = None,
        stream_prefix: str = None,
        stream_suffix: str = None,
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
            method=method,
            path=_path,
            mime_type=mime_type,
            chunk_delimiter=chunk_delimiter,
            chunk_prefix=chunk_prefix,
            chunk_suffix=chunk_suffix,
            stream_prefix=stream_prefix,
            stream_suffix=stream_suffix,
            extra_info=kwargs
        ))
        return fn

    return _api


def get(
        path: Optional[str] = None,
        mime_type: str = "application/json",
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
        chunk_suffix: str = None,
        stream_prefix: str = None,
        stream_suffix: str = None,
        **kwargs
) -> Callable:
    return api(
        method="GET",
        path=path,
        mime_type=mime_type,
        chunk_delimiter=chunk_delimiter,
        chunk_prefix=chunk_prefix,
        chunk_suffix=chunk_suffix,
        stream_prefix=stream_prefix,
        stream_suffix=stream_suffix,
        **kwargs
    )


def post(
        path: Optional[str] = None,
        mime_type: str = "application/json",
        chunk_delimiter: str = "\n\n",
        chunk_prefix: str = None,
        chunk_suffix: str = None,
        stream_prefix: str = None,
        stream_suffix: str = None,
        **kwargs
) -> Callable:
    return api(
        method="POST",
        path=path,
        mime_type=mime_type,
        chunk_delimiter=chunk_delimiter,
        chunk_prefix=chunk_prefix,
        chunk_suffix=chunk_suffix,
        stream_prefix=stream_prefix,
        stream_suffix=stream_suffix,
        **kwargs
    )


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


def _load_json_or_str(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


class ServiceError(RuntimeError):

    def __init__(self, text: str):
        err = _load_json_or_str(text)
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


class APIClient:

    def __init__(
            self,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            accept: str = "application/json",
            content_type: str = "application/json",
            user_agent: str = "API Client",
            connection: str = "close",
            verify=False,
            **extra_headers
    ) -> None:
        self.base_url = base_url
        self.headers = {
            "Accept": accept,
            "Content-Type": content_type,
            "User-Agent": user_agent,
            "Connection": connection,
            **extra_headers
        }
        if api_key is not None:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.verify = verify
        self.charset = "UTF-8"

    DEFAULT_CONN_POOL_SIZE = 10
    CONN_POOL = (
        PoolManager(DEFAULT_CONN_POOL_SIZE),
        PoolManager(DEFAULT_CONN_POOL_SIZE, cert_reqs='CERT_NONE')
    )

    def _request(
            self,
            method: Literal["GET", "POST"],
            url: str,
            body: Optional[str] = None,
            headers: Optional[Mapping[str, str]] = None,
            stream: bool = False,
            num_trials: int = 5,
            timeout: float = 15,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            verify: Optional[bool] = None,
    ):
        headers = self.headers if headers is None else headers
        verify = self.verify if verify is None else verify
        preload_content = not stream

        pool = self.CONN_POOL[int(not verify)]
        err = None
        for i in range(num_trials):
            try:
                return pool.request(
                    method=method,
                    url=url,
                    body=body,
                    headers=headers,
                    timeout=timeout * (1 + i * retry_factor),
                    preload_content=preload_content
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

    def get(
            self,
            path: Optional[str] = None, *,
            num_trials: int = 5,
            timeout: float = 15,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None
    ):
        full_url = urljoin(self.base_url, path)
        response = self._request(
            method="GET",
            url=full_url,
            num_trials=num_trials,
            timeout=timeout,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error
        )

        if response.status != 200:
            text = response.data.decode(self.charset)
            response.release_conn()
            raise ServiceError(text)

        try:
            return _load_json_or_str(response.data.decode(self.charset))
        finally:
            response.release_conn()

    def post(
            self,
            path: Optional[str] = None,
            json_data: Optional[Mapping] = None, *,
            stream: bool = False,
            exhaust_stream: bool = False,
            num_trials: int = 5,
            timeout: float = 15,
            interval: float = 1,
            retry_factor: float = 0.5,
            on_error: Optional[ErrorCallback] = None,
            chunk_delimiter: str = "\n\n",
            chunk_prefix: str = None,
            chunk_suffix: str = None,
            error_prefix: str = "ERROR: ",
            stream_read_size: int = 512
    ):
        full_url = urljoin(self.base_url, path)

        headers = {**self.headers}
        headers["Accept"] = headers["Accept"] + f"; stream={int(stream)}"
        body = json.dumps(json_data) if json_data is not None else None
        response = self._request(
            "POST",
            url=full_url,
            body=body,
            headers=headers,
            stream=stream,
            num_trials=num_trials,
            timeout=timeout,
            interval=interval,
            retry_factor=retry_factor,
            on_error=on_error
        )
        if response.status != 200:
            text = response.data.decode(self.charset)
            response.release_conn()
            raise ServiceError(text)

        if not stream:
            try:
                return _load_json_or_str(response.data.decode(self.charset))
            finally:
                response.release_conn()
        else:
            def iter_content():
                try:
                    if hasattr(response, "stream"):
                        yield from response.stream(stream_read_size, decode_content=True)
                    else:
                        while True:
                            data = response.read(stream_read_size)
                            if not data:
                                break
                            yield data
                finally:
                    response.release_conn()

            def iter_lines(contents: Iterable[bytes]) -> Iterable[str]:
                delimiter = chunk_delimiter.encode(self.charset) if chunk_delimiter is not None else None
                pending = None
                for data in contents:
                    if pending is not None:
                        data = pending + data

                    lines = data.split(delimiter) if delimiter else data.splitlines()
                    pending = lines.pop() if lines and lines[-1] and lines[-1][-1] == data[-1] else None

                    for line in lines:
                        yield line.decode(self.charset)

                if pending is not None:
                    yield pending.decode(self.charset)

            def iter_chunks(lines: Iterable[str]):
                error = None
                for chunk in lines:
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

            gen = iter_content()
            gen = iter_lines(gen)
            gen = iter_chunks(gen)
            return gen if not exhaust_stream else [*gen]
