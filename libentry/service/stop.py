#!/usr/bin/env python3

__author__ = "xi"

import os

from libentry import ArgumentParser
from libentry.service.running_utils import RunningConfig, RunningStatus


def main():
    parser = ArgumentParser()
    parser.add_argument("config_file", nargs="?")
    parser.add_argument("--dir_path", "-d")
    args = parser.parse_args()

    if args.config_file is None:
        return 0

    status = RunningStatus.load(args.config_file, dir_path=args.dir_path)
    if status is None:
        config = RunningConfig.load(args.config_file)
        if config is None:
            print(f"No such service \"{args.config_file}\".")
            return -1
        args.config_file = os.path.abspath(args.config_file)
        os.chdir(os.path.abspath(os.path.dirname(args.config_file)))
        status = RunningStatus.load(config.name, dir_path=args.dir_path)

    if status is None:
        print("No running service.")
        return 0

    if status.is_running():
        status.kill()
        print(f"Service \"{status.config.name}\" at process {status.pid} is stopped.")
    else:
        print(f"Process {status.pid} is not a running process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
