#!/usr/bin/env python3

__author__ = "xi"

from libentry import ArgumentParser
from libentry.service.running_utils import RunningStatus


def main():
    parser = ArgumentParser()
    parser.add_argument("--dir_path", "-d")
    args = parser.parse_args()

    service_dict = {}
    for status in RunningStatus.list(dir_path=args.dir_path):
        service_dict[status.config.name] = status

    for name, status in service_dict.items():
        if status.pid is None:
            running = "Not Running"
        elif status.is_running():
            running = f"Running({status.pid}) {status.config_path}"
        else:
            running = f"Stopped({status.pid}) {status.config_path}"
        print(f"{name}: {running}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
