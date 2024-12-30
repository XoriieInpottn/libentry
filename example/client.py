#!/usr/bin/env python3

__author__ = "xi"

from libentry.api import APIClient


def main():
    request = {"request_id": "test_request"}
    response = APIClient().post("http://localhost:3333/foo_stream", request, stream=True)
    for chunk in response:
        print(chunk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
