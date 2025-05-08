#!/usr/bin/env python3

__author__ = "xi"

from time import sleep
from typing import Iterable, Union

from pydantic import BaseModel

from libentry.mcp import api
from libentry.mcp.service import run_service


class AddParams(BaseModel):
    a: Union[int, float]
    b: Union[int, float] = 1


class Service:

    @api.tool()
    def add(self, params: AddParams) -> str:
        """Add two numbers."""
        # sleep(0.2)
        a = params.a
        b = params.b
        result = f"{a} + {b} = {a + b}"
        return result

    @api.api(tag="tool")
    def add_bundle(self, a: int, b: int) -> str:
        # sleep(0.2)
        return f"{a} + {b} = {a + b}"

    @api.resource("config://app")
    def config_file(self):
        """Config file of this app"""
        return b"hello"

    @api.tool()
    def add_stream(self, a: int, b: int) -> Iterable[str]:
        sleep(0.5)
        yield f"Step 1: a = {a}"
        sleep(0.5)
        yield f"Step 2: b = {b}"
        sleep(0.5)
        yield f"Answer: {a} + {b} = {a + b}"


def main():
    run_service(
        Service,
        host="127.0.0.1",
        port=8000
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
