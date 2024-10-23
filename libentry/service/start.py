#!/usr/bin/env python3

__author__ = "xi"

import os

from libentry import ArgumentParser
from libentry.service.running import Running


def main():
    parser = ArgumentParser()
    parser.add_argument("service_name", nargs="?")
    parser.add_argument("--base_dir", "-d", default=".")
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()

    service_name = args.service_name
    if service_name is None:
        return 0

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"\"{base_dir}\" does not exist.")
        return -1
    os.chdir(base_dir)
    run_dir = args.run_dir
    if run_dir is not None:
        run_dir = os.path.abspath(run_dir)

    running = Running.from_name(service_name, dir_path=run_dir)
    if running is not None and running.is_running():
        print(f"Service \"{running.name}\" is already running.")
        return -1

    running = Running.from_name(service_name, dir_path=base_dir)
    if running is None:
        print(f"Service \"{service_name}\" is not found.")
        return -1

    running.start(run_dir)
    print(f"Service \"{running.name}\" is started at process {running.pid}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
