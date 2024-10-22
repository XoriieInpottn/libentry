#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "RunningConfig",
    "RunningStatus",
]

import os.path
import tempfile
from time import sleep
from typing import Dict, List, Optional

import psutil
import yaml
from pydantic import BaseModel, Field, ValidationError


class RunningConfig(BaseModel):
    name: str = Field()
    exec: str = Field()
    envs: Dict[str, str] = Field(default_factory=dict)
    stdout: Optional[str] = Field(default="-")
    stderr: Optional[str] = Field(default="-")
    desc: str = Field(default="")

    @staticmethod
    def list(dir_path: str) -> List["RunningConfig"]:
        output_list = []
        for filename in os.listdir(dir_path):
            if not filename.endswith(".yml"):
                continue
            path = os.path.join(dir_path, filename)
            with open(path, "r") as f:
                try:
                    config = RunningConfig.model_validate(yaml.safe_load(f))
                except (yaml.YAMLError, ValidationError):
                    continue
                output_list.append(config)
        return output_list

    @staticmethod
    def load(config_file: str) -> Optional["RunningConfig"]:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                try:
                    return RunningConfig.model_validate(yaml.safe_load(f))
                except (yaml.YAMLError, ValidationError):
                    return None
        return None


class RunningStatus(BaseModel):
    config_path: str = Field()
    config: RunningConfig = Field()
    pid: int = Field()

    @staticmethod
    def list(dir_path: str = None) -> List["RunningStatus"]:
        if dir_path is None:
            dir_path = tempfile.gettempdir()

        output_list = []
        for filename in os.listdir(dir_path):
            if not filename.startswith("libentry."):
                continue
            path = os.path.join(dir_path, filename)
            with open(path, "r") as f:
                output_list.append(RunningStatus.model_validate(yaml.safe_load(f)))
        return output_list

    @staticmethod
    def load(name: str, dir_path: str = None) -> Optional["RunningStatus"]:
        if dir_path is None:
            dir_path = tempfile.gettempdir()
        path = os.path.join(dir_path, f"libentry.{name}")
        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    return RunningStatus.model_validate(yaml.safe_load(f))
                except (yaml.YAMLError, ValidationError):
                    return None
        return None

    def save(self, dir_path: str = None):
        if dir_path is None:
            dir_path = tempfile.gettempdir()
        path = os.path.join(dir_path, f"libentry.{self.config.name}")
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f)

    def remove(self, dir_path: str = None):
        if dir_path is None:
            dir_path = tempfile.gettempdir()
        path = os.path.join(dir_path, f"libentry.{self.config.name}")
        os.remove(path)

    def is_running(self):
        try:
            return psutil.Process(self.pid).is_running()
        except psutil.NoSuchProcess:
            return False

    def kill(self):
        proc = psutil.Process(self.pid)

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
