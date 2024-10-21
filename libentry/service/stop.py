#!/usr/bin/env python3

__author__ = "xi"

import os
from time import sleep

import psutil
import yaml

from libentry import ArgumentParser, logger
from libentry.service.common import RunningConfig
from libentry.service.running_utils import load_running, remove_running


def is_running(pid: int):
    try:
        return psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False


def kill_all(pid: int):
    proc = psutil.Process(pid)

    children = []
    for pid in psutil.pids():
        try:
            child = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue
        if child.ppid() == proc.pid:
            children.append(child)
    children.sort(key=lambda p: p.pid)

    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            continue
        while child.is_running():
            sleep(1)
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass
    while proc.is_running():
        sleep(1)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_file", "-f", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        logger.fatal(f"Config file \"{args.config_file}\" does not exist.")
        return -1
    with open(args.config_file) as f:
        config = RunningConfig.model_validate(yaml.safe_load(f))
    os.chdir(os.path.abspath(os.path.dirname(args.config_file)))

    status = load_running(config.name)
    if status is None:
        logger.info("No running service.")
        return 0

    if is_running(status.pid):
        kill_all(status.pid)
        remove_running(status.config.name)
        logger.info(f"Service \"{status.config.name}\" at process {status.pid} is stopped.")
    else:
        logger.error(f"Process {status.pid} is not a running process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
