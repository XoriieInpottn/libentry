#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "RunningConfig",
    "RunningStatus",
    "load_running",
    "save_running",
    "remove_running",
    "get_running_list",
]

import os.path
import tempfile
from typing import Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field


class RunningConfig(BaseModel):
    name: str = Field()
    exec: str = Field()
    envs: Dict[str, str] = Field(default_factory=dict)
    stdout: Optional[str] = Field(default="-")
    stderr: Optional[str] = Field(default="-")
    desc: str = Field(default="")


class RunningStatus(BaseModel):
    config_path: str = Field()
    config: RunningConfig = Field()
    pid: int = Field()


def load_running(name: str) -> Optional[RunningStatus]:
    path = os.path.join(tempfile.gettempdir(), f"libentry.{name}")
    if os.path.exists(path):
        with open(path, "r") as f:
            return RunningStatus.model_validate(yaml.safe_load(f))
    return None


def save_running(status: RunningStatus):
    path = os.path.join(tempfile.gettempdir(), f"libentry.{status.config.name}")
    with open(path, "w") as f:
        yaml.safe_dump(status.model_dump(), f)


def remove_running(name: Union[str, RunningStatus]):
    if isinstance(name, RunningStatus):
        name = name.config.name
    path = os.path.join(tempfile.gettempdir(), f"libentry.{name}")
    os.remove(path)


def get_running_list():
    return [
        filename[len("libentry."):]
        for filename in os.listdir(tempfile.gettempdir())
        if filename.startswith("libentry.")
    ]
