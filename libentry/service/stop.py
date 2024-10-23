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
    if running is None:
        print("No running service.")
        return 0

    if running.is_running():
        running.kill()
        print(f"Service \"{running.name}\" at process {running.pid} is stopped.")
    else:
        print(f"Process {running.pid} is not a running process.")
    running.remove(dir_path=run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
