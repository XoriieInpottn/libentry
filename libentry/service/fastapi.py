#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "run_service",
]

import asyncio
from functools import update_wrapper
from types import GeneratorType

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, StreamingResponse

from .common import JSONDumper
from ..api import list_api_info
from ..logging import logger


def run_service(service, host: str, port: int):
    logger.info("Parsing APIs")
    api_info_list = list_api_info(service)
    if len(api_info_list) == 0:
        logger.error("No API found, nothing to serve.")
        return

    app = FastAPI()
    for fn, api_info in api_info_list:
        method = api_info.method
        path = api_info.path
        is_async = asyncio.iscoroutinefunction(fn)
        fn_type = "Async-" if is_async else ""
        logger.info(f"Serve {fn_type}API-{method} for {path}")

        dumper = JSONDumper(api_info)
        if is_async:
            async def _fastapi_wrapper(*args, **kwargs):
                response = await fn(*args, **kwargs)

                if isinstance(response, (GeneratorType, range)):
                    return StreamingResponse(
                        dumper.dump_stream(response),
                        media_type=api_info.mime_type
                    )
                else:
                    return PlainTextResponse(
                        dumper.dump(response),
                        media_type=api_info.mime_type
                    )
        else:
            def _fastapi_wrapper(*args, **kwargs):
                response = fn(*args, **kwargs)

                if isinstance(response, (GeneratorType, range)):
                    return StreamingResponse(
                        dumper.dump_stream(response),
                        media_type=api_info.mime_type
                    )
                else:
                    return PlainTextResponse(
                        dumper.dump(response),
                        media_type=api_info.mime_type
                    )

        update_wrapper(_fastapi_wrapper, fn)

        if method == "GET":
            app.get(path)(_fastapi_wrapper)
        elif method == "POST":
            app.post(path)(_fastapi_wrapper)
        else:
            raise RuntimeError(f"Unsupported method \"{method}\" for ")

    logger.info("Starting server")
    uvicorn.run(app, host=host, port=port)
