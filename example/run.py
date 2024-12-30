#!/usr/bin/env python3

__author__ = "xi"

from example.common import ExampleServiceConfig
from example.service import ExampleService
from libentry.service.flask import run_service


def main():
    run_service(
        service_type=ExampleService,
        service_config=ExampleServiceConfig(name="MyExampleService"),
        host="0.0.0.0",
        port=3333,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
