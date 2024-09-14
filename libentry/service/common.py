#!/usr/bin/env python3

__all__ = [
    "JSONDumper"
]

from typing import Iterable

from pydantic import BaseModel

from .. import json
from ..api import APIInfo


class JSONDumper:

    def __init__(self, api_info: APIInfo):
        self.api_info = api_info

    def dump_stream(self, response: Iterable):
        if self.api_info.stream_prefix is not None:
            yield self.api_info.stream_prefix

        if self.api_info.chunk_delimiter is not None:
            yield self.api_info.chunk_delimiter

        for item in response:
            text = self.dump(item)

            if self.api_info.chunk_prefix is not None:
                yield self.api_info.chunk_prefix

            yield text

            if self.api_info.chunk_suffix is not None:
                yield self.api_info.chunk_suffix

            if self.api_info.chunk_delimiter is not None:
                yield self.api_info.chunk_delimiter

        if self.api_info.stream_suffix is not None:
            yield self.api_info.stream_suffix

        if self.api_info.chunk_delimiter is not None:
            yield self.api_info.chunk_delimiter

    @staticmethod
    def dump(response) -> str:
        if response is None:
            return ""
        elif isinstance(response, BaseModel):
            return json.dumps(response.model_dump())
        else:
            try:
                return json.dumps(response)
            except TypeError:
                return repr(response)
