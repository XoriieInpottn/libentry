#!/usr/bin/env python3

__author__ = "xi"

from types import GeneratorType
from typing import List

import rich

from libentry.mcp.client import APIClient


def print_response(resp):
    if not isinstance(resp, (GeneratorType, List)):
        rich.print(resp)
    else:
        for item in resp:
            rich.print(item)
    print()


def main():
    client = APIClient("http://localhost:8000")

    print("Subroutine\n" + "=" * 80)
    resp = client.post("/add", {"a": 1, "b": 2})
    print_response(resp)

    print("Subroutine stream\n" + "=" * 80)
    resp = client.post("/add_stream", {"a": 1, "b": 2})
    print_response(resp)

    ################################################################################

    print("JSONRPC\n" + "=" * 80)
    resp = client.call("add", {"a": 1, "b": 2})
    print_response(resp)

    print("JSONRPC stream\n" + "=" * 80)
    resp = client.call("add_stream", {"a": 1, "b": 2})
    print_response(resp)

    print("JSONRPC tools/list\n" + "=" * 80)
    resp = client.list_tools().tools
    print_response(resp)

    print("JSONRPC tools/call\n" + "=" * 80)
    resp = client.call_tool("add", {"a": 1, "b": 2})
    print_response(resp)

    print("JSONRPC tools/call stream\n" + "=" * 80)
    resp = client.call_tool("add_stream", {"a": 1, "b": 2})
    print_response(resp)

    print("JSONRPC resources/list\n" + "=" * 80)
    resp = client.list_resources().resources
    print_response(resp)

    print("JSONRPC resources/read\n" + "=" * 80)
    resp = client.read_resource("config://app")
    print_response(resp)

    ################################################################################

    print("Session initialize\n" + "=" * 80)
    session = client.start_session()
    resp = session.initialize()
    print_response(resp)

    print("Session tools/list\n" + "=" * 80)
    resp = session.list_tools().tools
    print_response(resp)

    print("Session tools/call\n" + "=" * 80)
    resp = session.call_tool("add", {"a": 1, "b": 2})
    print_response(resp)

    print("Session tools/call stream\n" + "=" * 80)
    resp = session.call_tool("add_stream", {"a": 1, "b": 2})
    print_response(resp)

    print("Session resources/list\n" + "=" * 80)
    resp = session.list_resources().resources
    print_response(resp)

    print("Session resources/read\n" + "=" * 80)
    resp = session.read_resource("config://app")
    print_response(resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
