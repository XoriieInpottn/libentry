#!/usr/bin/env python3

__author__ = "xi"

from types import GeneratorType

from example.common import ExampleRequest, ExampleResponse
from libentry.mcp.client import APIClient


def main():
    client = APIClient("http://localhost:8888")

    request = ExampleRequest(stream=True)
    response = client.post(request)
    assert isinstance(response, GeneratorType)
    it = iter(response)
    try:
        while True:
            chunk = next(it)
            chunk = ExampleResponse.model_validate(chunk)
            print(chunk.output_content, end="", flush=True)
    except StopIteration as e:
        print()
        if e.value is not None:
            chunk = ExampleResponse.model_validate(e.value)
            print(f"完整输出：\b{chunk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
