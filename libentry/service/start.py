#!/usr/bin/env python3

__author__ = "xi"

import os
import shlex
import subprocess

import psutil
import yaml

from libentry import ArgumentParser, logger
from libentry.service.common import RunningConfig, RunningStatus
from libentry.service.running_utils import load_running, save_running


def is_running(pid: int):
    try:
        return psutil.Process(pid).is_running()
    except psutil.NoSuchProcess:
        return False


def main():
    parser = ArgumentParser()
    parser.add_argument("--config_file", "-f", required=True)
    parser.add_argument("--shell", default="/bin/bash -c")
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        logger.fatal(f"Config file \"{args.config_file}\" does not exist.")
        return -1
    with open(args.config_file) as f:
        config = RunningConfig.model_validate(yaml.safe_load(f))
    os.chdir(os.path.abspath(os.path.dirname(args.config_file)))

    status = load_running(config.name)
    if status is not None and is_running(status.pid):
        logger.fatal(f"Service \"{status.config.name}\" is already running.")
        return -1

    envs = None
    if config.envs:
        envs = {**os.environ, **config.envs}

    if config.stdout == "-":
        stdout = None
    elif config.stdout is None:
        stdout = subprocess.DEVNULL
    else:
        stdout = open(config.stdout, "a")

    if config.stderr == "-":
        stderr = None
    elif config.stderr is None:
        stderr = subprocess.DEVNULL
    else:
        stderr = open(config.stderr, "a")

    logger.info(f"Starting service \"{config.name}\".")
    process = subprocess.Popen(
        [*shlex.split(args.shell), config.exec],
        cwd=os.getcwd(),
        env=envs,
        preexec_fn=os.setpgrp,
        stdout=stdout,
        stderr=stderr
    )
    pid = process.pid
    pgid = os.getpgid(pid)
    logger.info(f"Service \"{config.name}\" started at process {pgid}.")

    status = RunningStatus(config_path=os.path.abspath(args.config_file), config=config, pid=pid)
    save_running(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
