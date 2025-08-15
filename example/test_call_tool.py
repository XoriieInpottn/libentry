#!/usr/bin/env python3

__author__ = "xi"

from types import GeneratorType

from example.common import ExampleRequest, ExampleResponse
from libentry.mcp.client import APIClient
from libentry.mcp.types import CallToolResult


def main():
    client = APIClient("http://localhost:8888")

    request = ExampleRequest(stream=True)
    response = client.call_tool("give_example", request.model_dump())
    assert isinstance(response, GeneratorType)
    it = iter(response)
    try:
        while True:
            chunk = next(it)
            assert isinstance(chunk, CallToolResult)
            chunk = ExampleResponse.model_validate(chunk.structuredContent)
            print(chunk.output_content, end="", flush=True)
    except StopIteration as e:
        print()
        if e.value is not None:
            chunk = e.value
            assert isinstance(chunk, CallToolResult)
            chunk = ExampleResponse.model_validate(chunk.structuredContent)
            print(f"完整输出：\n{chunk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
