#!/usr/bin/env python3

__author__ = "xi"

from libentry.mcp.client import APIClient, SSESession
from libentry.mcp.types import JSONRPCRequest


def main():
    client = APIClient("http://localhost:8000")

    print("Subroutine\n" + "=" * 80)
    resp = client.subroutine_request("/add", {"a": 1, "b": 2})
    print(resp.result)
    print()

    print("Subroutine stream\n" + "=" * 80)
    resp = client.subroutine_request("/add_stream", {"a": 1, "b": 2})
    for item in resp:
        print(item.result, flush=True)
    print()

    ################################################################################

    print("RPC\n" + "=" * 80)
    resp = client.rpc_request(JSONRPCRequest(jsonrpc="2.0", method="add", id=1, params={"a": 1, "b": 2}))
    print(resp.result)
    print()

    print("RPC stream\n" + "=" * 80)
    resp = client.rpc_request(JSONRPCRequest(jsonrpc="2.0", method="add_stream", id=1, params={"a": 1, "b": 2}))
    for item in resp:
        print(item.result, flush=True)
    print()

    print("RPC tools/list\n" + "=" * 80)
    for tool in client.list_tools().tools:
        print(tool.model_dump_json(exclude_none=True))
    print()

    print("RPC tools/call\n" + "=" * 80)
    resp = client.call_tool("add", {"a": 1, "b": 2})
    print(resp)
    print()

    print("RPC resources/list\n" + "=" * 80)
    for resource in client.list_resources().resources:
        print(resource.model_dump_json(exclude_none=True))
    print()

    print("RPC resources/read\n" + "=" * 80)
    resp = client.read_resource("config://app")
    print(resp)
    print()

    ################################################################################

    print("Session initialize\n" + "=" * 80)
    session = SSESession(client)
    print(session.initialize())
    print()

    print("Session tools/list\n" + "=" * 80)
    for tool in session.list_tools().tools:
        print(tool.model_dump_json(exclude_none=True))
    print()

    print("Session tools/call\n" + "=" * 80)
    resp = session.call_tool("add", {"a": 1, "b": 2})
    print(resp)
    print()

    print("Session resources/list\n" + "=" * 80)
    for resource in session.list_resources().resources:
        print(resource.model_dump_json(exclude_none=True))
    print()

    print("Session resources/read\n" + "=" * 80)
    resp = session.read_resource("config://app")
    print(resp)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
