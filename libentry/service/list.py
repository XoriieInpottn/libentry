#!/usr/bin/env python3

__author__ = "xi"

import os

from libentry import ArgumentParser
from libentry.service.running import Running


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_dir", "-d", default=".")
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        print(f"\"{base_dir}\" does not exist.")
        return -1
    os.chdir(base_dir)
    run_dir = args.run_dir
    if run_dir is not None:
        run_dir = os.path.abspath(run_dir)

    service_dict = {}
    for running in Running.list(base_dir):
        service_dict[running.name] = running

    if run_dir is None or os.path.exists(run_dir):
        for running in Running.list(run_dir):
            service_dict[running.name] = running

    for name, value in service_dict.items():
        if value.exec is None:
            running = "Symbolic"
            metadata = ""
        elif value.pid is None:
            running = "Stopped"
            metadata = ""
        elif value.is_running():
            running = f"Running({value.pid})"
            metadata = value.metadata
        else:
            running = f"Stopped({value.pid})"
            metadata = ""

        print(f"{name}\t{running}\t{metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
