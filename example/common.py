#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ExampleServiceConfig",
    "ExampleRequest",
    "ExampleResponse",
]

import os

from agent_types.common import Request, Response
from pydantic import BaseModel, Field


class ExampleServiceConfig(BaseModel):
    name: str = Field("default_example")
    other_configs: dict = Field(default_factory=dict)


class ExampleRequest(Request):
    __request_name__ = "foo_stream"

    path: str = Field(default=os.path.abspath(__file__))
    stream: bool = Field(True)


class ExampleResponse(Response):
    output_content: str = Field()
