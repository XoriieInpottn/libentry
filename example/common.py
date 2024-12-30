#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "ExampleServiceConfig",
    "ExampleRequest",
    "ExampleResponse",
]

from pydantic import BaseModel, Field


class ExampleServiceConfig(BaseModel):
    name: str = Field("default_example")
    other_configs: dict = Field(default_factory=dict)



class ExampleRequest(BaseModel):
    request_id: str = Field(min_length=10, max_length=1024)
    stream: bool = Field(True)


class ExampleResponse(BaseModel):
    output_content: str = Field()
