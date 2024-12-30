#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ExampleService",
]

import random
from time import sleep
from typing import Iterable, Union

from example.common import ExampleRequest, ExampleResponse, ExampleServiceConfig
from libentry import api, logger


class ExampleService:

    def __init__(self, config: ExampleServiceConfig):
        self.config = config

    @api.post()
    def foo(self, request: ExampleRequest) -> ExampleResponse:
        logger.info(f"{request.request_id}: step A")
        sleep(random.uniform(1, 1.1))

        logger.info(f"{request.request_id}: step B")
        sleep(random.uniform(1, 1.2))

        logger.info(f"{request.request_id}: step C")
        sleep(random.uniform(1, 1.1))

        logger.info(f"{request.request_id}: finished")
        return ExampleResponse(output_content=f"{self.config.name} finished")

    @api.post()
    def foo_stream(self, request: ExampleRequest) -> Union[ExampleResponse, Iterable[ExampleResponse]]:
        if request.stream:
            def gen():
                try:
                    logger.info(f"{request.request_id}: step A")
                    yield ExampleResponse(output_content=f"step A")
                    sleep(random.uniform(1, 1.1))

                    logger.info(f"{request.request_id}: step B")
                    yield ExampleResponse(output_content=f"step B")
                    sleep(random.uniform(1, 1.2))

                    logger.info(f"{request.request_id}: step C")
                    yield ExampleResponse(output_content=f"step C")
                    sleep(random.uniform(1, 1.1))

                    logger.info(f"{request.request_id}: finished")
                    return ExampleResponse(output_content=f"{self.config.name} finished(stream)")
                finally:
                    logger.info(f"{request.request_id}: stream released")

            return gen()
        else:
            return ExampleResponse(output_content=f"{self.config.name} finished(no_stream)")
