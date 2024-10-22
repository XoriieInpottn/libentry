#!/usr/bin/env python3

__author__ = "xi"

import os
import shlex
import subprocess

from libentry import ArgumentParser, logger
from libentry.service.running_utils import RunningConfig, RunningStatus


def main():
    parser = ArgumentParser()
    parser.add_argument("config_file", nargs="?")
    parser.add_argument("--dir_path", "-d")
    parser.add_argument("--shell", default="/bin/bash -c")
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
    else:
        config = status.config
        os.chdir(os.path.abspath(os.path.dirname(status.config_path)))

    if status is not None and status.is_running():
        print(f"Service \"{status.config.name}\" is already running.")
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

    if status is None:
        status = RunningStatus(
            config_path=os.path.abspath(args.config_file),
            config=config,
            pid=pid
        )
    else:
        status.pid = pid
    status.save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
