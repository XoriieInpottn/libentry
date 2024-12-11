#!/usr/bin/env python3

__author__ = "xi"

import random
from time import sleep
from typing import Iterable, Union

from pydantic import BaseModel, Field

from libentry import api, logger
from libentry.service.flask import run_service


class ExampleRequest(BaseModel):
    request_id: str = Field()
    stream: bool = Field(True)


class ExampleResponse(BaseModel):
    output_content: str = Field()


class ExampleService:

    @api.post()
    def foo(self, request: ExampleRequest) -> ExampleResponse:
        logger.info(f"{request.request_id}: step A")
        sleep(random.uniform(1, 1.1))

        logger.info(f"{request.request_id}: step B")
        sleep(random.uniform(1, 1.2))

        logger.info(f"{request.request_id}: step C")
        sleep(random.uniform(1, 1.1))

        logger.info(f"{request.request_id}: finished")
        return ExampleResponse(output_content=f"finished")

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
                    return ExampleResponse(output_content=f"finished(stream)")
                finally:
                    logger.info(f"{request.request_id}: stream released")

            return gen()
        else:
            return ExampleResponse(output_content=f"finished(no_stream)")

    @api.get()
    def foo1(self, request_id: str, stream: bool = False) -> str:
        sleep(1)
        return "finished"


def main():
    run_service(
        service_type=ExampleService,
        service_config=None,
        host="0.0.0.0",
        port=3333,
        num_workers=1,
        num_threads=50,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
