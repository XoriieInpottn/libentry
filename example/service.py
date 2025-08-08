#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ExampleService",
]

import random
import time
from typing import Generator, Iterable, Optional, Union

from example.common import ExampleRequest, ExampleResponse, ExampleServiceConfig
from libentry import logger
from libentry.mcp import api


class ExampleService:

    def __init__(self, config: ExampleServiceConfig):
        self.config = config

    @api.tool(path=ExampleRequest.get_request_path())
    def foo_stream(self, request: ExampleRequest) -> Union[ExampleResponse, Iterable[ExampleResponse]]:
        chars_stream = self._iter_chars(request.path)
        if request.stream:
            return self._iter_example_response(chars_stream, request)
        else:
            return ExampleResponse(output_content="".join(chars_stream))


    def _iter_example_response(
            self,
            chars_stream: Iterable[str],
            request: ExampleRequest
    ) -> Generator[ExampleResponse, None, Optional[ExampleResponse]]:
        try:
            logger.info(f"{request.trace_id}: Start {request.path}")
            yield ExampleResponse(output_content=f"Start {request.path}\n")

            full = []
            for chunk in chars_stream:
                yield ExampleResponse(output_content=chunk)
                full.append(chunk)

            logger.info(f"{request.trace_id}: End {request.path}")
            yield ExampleResponse(output_content=f"End {request.path}\n")

            return ExampleResponse(output_content="".join(full))
        finally:
            logger.info(f"{request.trace_id}: stream released")

    @staticmethod
    def _iter_chars(path: str):
        with open(path, "r") as f:
            while True:
                time.sleep(random.uniform(0.01, 0.05))
                chunk = f.read(random.randint(4, 6))
                if not chunk:
                    break
                yield chunk
