#!/usr/bin/env python3

__author__ = "xi"

import json
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import time
from typing import Literal, Optional

from pydantic import BaseModel

from libentry.mcp.client import APIClient
from libentry.mcp.types import HTTPOptions


class TestRequest(BaseModel):
    url: str
    method: Literal["GET", "POST"] = "GET"
    data: Optional[dict] = None
    timeout: float = 15
    num_threads: int = 1
    num_calls: int = 1
    stream: bool = False
    quiet: bool = False
    max_resp_len: int = 100


class TestResponse(BaseModel):
    num_success: int = 0
    num_failed: int = 0
    avg_time: float = 0
    max_time: float = 0
    min_time: float = 0


def test(request: TestRequest):
    time_list = []
    lock = Lock()

    def _worker(tid):
        for cid in range(request.num_calls):
            try:
                kwargs = dict(
                    on_error=lambda err: print(f"[{tid}:{cid}:RETRY] {err}"),
                    timeout=request.timeout
                )
                t = time()
                if request.method == "GET":
                    response = APIClient().get(
                        request.url,
                        HTTPOptions(**kwargs)
                    )
                else:
                    response = APIClient().post(
                        request.url,
                        request.data,
                        HTTPOptions(stream=request.stream, **kwargs)
                    )
                if not request.stream:
                    t = time() - t
                    if not request.quiet:
                        content = str(response).replace("\n", "\\n")
                        print(f"[{tid}:{cid}:SUCCESS] Response={content:.{request.max_resp_len}} Time={t:.04f}")
                else:
                    if not request.quiet and request.num_threads == 1:
                        print(f"[{tid}:{cid}:SUCCESS] Response=[", end="", flush=True)
                        for chunk in response:
                            print(f"'{chunk}', ", end="", flush=True)
                        t = time() - t
                        print(f"] Time={t:.04f}", flush=True)
                    else:
                        content = [*response]
                        t = time() - t
                        if not request.quiet:
                            content = str(content).replace("\n", "\\n")
                            print(f"[{tid}:{cid}:SUCCESS] Response={content:.{request.max_resp_len}} Time={t:.04f}")
                with lock:
                    time_list.append(t)
            except Exception as e:
                print(f"[{tid}:{cid}:FAILED] {e}")

    with ThreadPoolExecutor(request.num_threads) as pool:
        futures = [pool.submit(_worker, i) for i in range(request.num_threads)]
        for future in futures:
            future.result()

    if len(time_list) > 0:
        return TestResponse(
            num_success=len(time_list),
            num_failed=request.num_threads * request.num_calls - len(time_list),
            max_time=max(time_list),
            min_time=min(time_list),
            avg_time=sum(time_list) / len(time_list)
        )
    else:
        return TestResponse(
            num_success=0,
            num_failed=request.num_threads * request.num_calls,
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("url", nargs="?")
    parser.add_argument("--method", "-m", default="GET", choices=["GET", "POST"])
    parser.add_argument("--data", "-d")
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--num_threads", "-t", type=int, default=1)
    parser.add_argument("--num_calls", "-c", type=int, default=1)
    parser.add_argument("--stream", "-s", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    if args.url is None:
        print("URL should be given.")
        return 1

    data = args.data
    if data is not None:
        data = json.loads(data)
    response = test(TestRequest(
        url=args.url,
        method=args.method,
        data=data,
        timeout=args.timeout,
        num_threads=args.num_threads,
        num_calls=args.num_calls,
        stream=args.stream,
        quiet=args.quiet
    ))
    print(response.model_dump_json(indent=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
