#!/usr/bin/env python3
from time import sleep
from typing import Iterable

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Create an MCP server
mcp = FastMCP("Demo")


# Add an addition tool
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


class AddRequest(BaseModel):
    a: int
    b: int


@mcp.tool()
def foo(request: AddRequest) -> str:
    sleep(0.5)
    a, b = request.a, request.b
    return f"{a} + {b} = {a + b}"


@mcp.tool()
def foo_stream(a: int, b: int) -> Iterable[str]:
    for i in range(3):
        sleep(0.5)
        yield f"{a} + {b} = {a + b}?{i}"


mcp.run("sse")
