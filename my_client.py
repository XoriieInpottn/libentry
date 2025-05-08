#!/usr/bin/env python3

__author__ = "xi"

from libentry.mcp.client import APIClient, SSESession
from libentry.mcp.types import JSONRPCRequest, JSONRequest


def main():
    client = APIClient("http://localhost:8000")

    for tool in client.list_tools().tools:
        print(tool.model_dump_json(exclude_none=True))
    for resource in client.list_resources().resources:
        print(resource.model_dump_json(exclude_none=True))
    print()

    resp = client.rpc_request(JSONRPCRequest(jsonrpc="2.0", method="add", id=1, params={"a": 1, "b": 2}))
    print(resp)
    print()

    resp = client.rpc_request(JSONRPCRequest(jsonrpc="2.0", method="add_stream", id=1, params={"a": 1, "b": 2}))
    for item in resp:
        print(item, flush=True)
    print()

    # resp = client.request(JSONRequest(method="POST", path="/add", json_obj={"a": 1, "b": 2}))
    # print(resp.content)
    # print()
    #
    # resp = client.request(JSONRequest(method="POST", path="/add_stream", json_obj={"a": 1, "b": 2}))
    # for item in resp.content:
    #     print(item, flush=True)
    # print()
    #
    # resp = client.rpc_request(
    #     JSONRPCRequest(jsonrpc="2.0", method="resources/read", id=1, params={"uri": "config://app"}))
    # print(resp)
    # print()

    session = SSESession(client)
    print(session.initialize())
    print(session.list_tools())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
