#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ExampleService",
]

import random
import time
from typing import Iterable, Union

from example.common import ExampleRequest, ExampleResponse, ExampleServiceConfig
from libentry import logger
from libentry.mcp import api


class ExampleService:

    def __init__(self, config: ExampleServiceConfig):
        self.config = config

    @api.tool(path=ExampleRequest.get_request_path())
    def foo_stream(self, request: ExampleRequest) -> Union[ExampleResponse, Iterable[ExampleResponse]]:
        if request.stream:
            def gen():
                try:
                    logger.info(f"{request.trace_id}: Start {request.path}")
                    yield ExampleResponse(output_content=f"Start {request.path}\n")

                    full = []
                    with open(request.path, "r") as f:
                        while True:
                            time.sleep(random.uniform(0.01, 0.05))
                            chunk = f.read(random.randint(4, 6))
                            if not chunk:
                                break
                            yield ExampleResponse(output_content=chunk)
                            full.append(chunk)

                    logger.info(f"{request.trace_id}: End {request.path}")
                    yield ExampleResponse(output_content=f"End {request.path}\n")

                    return ExampleResponse(output_content="".join(full))
                finally:
                    logger.info(f"{request.trace_id}: stream released")

            return gen()
        else:
            with open(request.path, "r") as f:
                return ExampleResponse(output_content=f.read())
