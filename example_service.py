#!/usr/bin/env python3

__author__ = "xi"

import random
from time import sleep
from typing import Iterable, List, Optional, Union

from pydantic import BaseModel, Field

from libentry import api, logger
from libentry.service.flask import run_service


class ExampleRequest(BaseModel):
    request_id: str = Field(min_length=10, max_length=1024)
    a: float = Field(gt=0, le=10)
    stream: bool = Field(True)


class ExampleResponse(BaseModel):
    output_content: str = Field()


class ToolInputParam(BaseModel):
    """ tool input parameters base model, arguments read from memory need to define 'memory_operator'
        e.g.:
        class MyInputParam(ToolInputParam):
            uid: str = Field(default=None, description="uid", is_required=True)
            query: str = Field(default=None, description="query", is_required=True)
            chat_history: str = Field(
                default=None, description="chat history", is_required=True, memory_operator="chat_history_reader"
            )
    """
    trace_id: Optional[str] = Field(default=None, description="trace id in different service")


class PreferenceQuestionToolInputParam(ToolInputParam):
    query: str = Field(..., description="用户问题")
    category: str = Field(..., description="商品类别")
    chat_history: str = Field(
        default="",
        description="chat_history",
    )
    user_preferences: List[str] = Field(
        default=[],
        description="用户偏好",
        example=["预算: 9000元", "使用场景及需求:打游戏"],
    )
    history_preferences_record: List[str] = Field(
        default=[],
        description="记录之前追问问过哪些方面的问题"
    )


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

    @api.post()
    def foo2(self, request_ids: List[str]) -> str:
        l = len(request_ids[0])
        return "finished" + str(l)

    @api.post()
    def foo3(self, request: PreferenceQuestionToolInputParam):
        sleep(1)
        return "finished"


def main():
    run_service(
        service_type=ExampleService,
        service_config=None,
        host="0.0.0.0",
        port=3333,
        num_workers=10,
        num_threads=50,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
