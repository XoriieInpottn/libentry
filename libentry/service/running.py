#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "Running",
]

import os.path
import shlex
import subprocess
import tempfile
from time import sleep
from typing import Dict, List, Optional

import psutil
import yaml
from pydantic import BaseModel, Field, ValidationError

DEFAULT_SUFFIX = ".run.yml"
DEFAULT_RUNNING_DIR = os.path.join(tempfile.gettempdir(), "libentry.running")


class Running(BaseModel):
    name: str = Field()
    metadata: dict = Field(default_factory=dict)
    exec: Optional[str] = Field(default=None)
    shell: str = Field(default="/bin/bash -c")
    envs: Dict[str, str] = Field(default_factory=dict)
    stdout: Optional[str] = Field(default=None)
    stderr: Optional[str] = Field(default=None)
    desc: Optional[str] = Field(default=None)

    pid: Optional[int] = Field(default=None)

    @staticmethod
    def list(dir_path: Optional[str] = None, ext: str = DEFAULT_SUFFIX) -> List["Running"]:
        if dir_path is None:
            dir_path = DEFAULT_RUNNING_DIR

        if not os.path.exists(dir_path):
            return []

        output_list = []
        for filename in os.listdir(dir_path):
            if not filename.endswith(ext):
                continue
            path = os.path.join(dir_path, filename)
            output = Running.from_file(path)
            if output is None:
                continue
            output_list.append(output)
        return output_list

    @staticmethod
    def from_name(name: str, dir_path: Optional[str] = None, ext: str = DEFAULT_SUFFIX) -> Optional["Running"]:
        if dir_path is None:
            dir_path = DEFAULT_RUNNING_DIR

        return Running.from_file(os.path.join(dir_path, name + ext))

    @staticmethod
    def from_file(path: str) -> Optional["Running"]:
        if os.path.exists(path):
            with open(path, "r") as f:
                return Running.from_string(f.read())
        return None

    @staticmethod
    def from_string(content: str) -> Optional["Running"]:
        try:
            json_obj = yaml.safe_load(content)
            model_obj = Running.model_validate(json_obj)
            return model_obj
        except (yaml.YAMLError, ValidationError) as e:
            print(e)
            return None

    def save(self, dir_path: Optional[str] = None, ext: str = DEFAULT_SUFFIX) -> str:
        if dir_path is None:
            dir_path = DEFAULT_RUNNING_DIR

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        content = yaml.safe_dump(self.model_dump())
        dir_path = os.path.join(dir_path, self.name + ext)
        with open(dir_path, "w") as f:
            f.write(content)
        return content

    def remove(self, dir_path: str = None, ext: str = DEFAULT_SUFFIX):
        if dir_path is None:
            dir_path = DEFAULT_RUNNING_DIR

        path = os.path.join(dir_path, self.name + ext)
        os.remove(path)

    def start(self, dir_path: Optional[str] = None):
        if self.exec is None:
            return

        envs = None
        if self.envs:
            envs = {**os.environ, **self.envs}

        if self.stdout == "-":
            stdout = None
        elif self.stdout is None:
            stdout = subprocess.DEVNULL
        else:
            stdout = open(self.stdout, "a")

        if self.stderr == "-":
            stderr = None
        elif self.stderr is None:
            stderr = subprocess.DEVNULL
        else:
            stderr = open(self.stderr, "a")

        process = subprocess.Popen(
            [*shlex.split(self.shell), self.exec],
            cwd=os.getcwd(),
            env=envs,
            preexec_fn=os.setpgrp,
            stdout=stdout,
            stderr=stderr
        )
        self.pid = process.pid

        self.save(dir_path)

    def is_running(self):
        if self.pid is None:
            return False
        try:
            return psutil.Process(self.pid).is_running()
        except psutil.NoSuchProcess:
            return False

    def kill(self):
        if self.pid is None:
            return

        cpid_list = get_children(self.pid, recursive=False)
        cpid_list.sort()
        for cpid in cpid_list:
            try:
                child = psutil.Process(cpid)
                child.kill()
                while child.is_running():
                    sleep(1)
            except psutil.NoSuchProcess:
                continue

        try:
            proc = psutil.Process(self.pid)
            proc.kill()
            while proc.is_running():
                sleep(1)
        except psutil.NoSuchProcess:
            pass


def get_children(pid: int, recursive=True, pids: List[int] = None) -> List[int]:
    if pids is None:
        pids = psutil.pids()

    output_list = []
    for _pid in pids:
        try:
            if psutil.Process(_pid).ppid() == pid:
                output_list.append(_pid)
        except psutil.NoSuchProcess:
            continue

    if recursive:
        cpid_list = [*output_list]
        for cpid in cpid_list:
            output_list += get_children(cpid, True, pids)
    return output_list
