#!/usr/bin/env python3

__author__ = "xi"

from example.common import ExampleServiceConfig
from example.service import ExampleService
from libentry import ArgumentParser
from libentry.mcp.service import RunServiceConfig, run_service


def main():
    parser = ArgumentParser()
    parser.add_schema("run", RunServiceConfig)
    parser.add_schema("config", ExampleServiceConfig)
    args = parser.parse_args()

    run_service(
        service_type=ExampleService,
        service_config=args.config,
        run_config=args.run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
