#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "AbstractResourceManager",
    "JSONResourceManager",
    "RemoteResourceManager",
    "ResourceManager",
]

import abc
from typing import Any, Dict, List
from urllib.parse import urlparse

import yaml


class AbstractResourceManager(abc.ABC):

    @abc.abstractmethod
    def list(self) -> List[str]:
        pass

    @abc.abstractmethod
    def get(self, name: str) -> Any:
        pass

    @abc.abstractmethod
    def set(self, name: str, value: Any) -> Any:
        pass


class JSONResourceManager(AbstractResourceManager):

    def __init__(self, path: str):
        with open(path, "r") as f:
            self.resource_obj = yaml.load(f, yaml.SafeLoader)

        if not isinstance(self.resource_obj, Dict):
            raise RuntimeError(
                f"Expect a JSON object, "
                f"got {type(self.resource_obj)}."
            )

    def list(self) -> List[str]:
        return [*self.resource_obj.keys()]

    def get(self, name: str) -> Any:
        return self.resource_obj.get(name)

    def set(self, name: str, value: Any) -> Any:
        raise NotImplementedError("\"set\" is not supported by JSONResourceClient.")


class RemoteResourceManager(AbstractResourceManager):

    def __init__(self, url: str):
        pass


class ResourceManager:

    def __new__(cls, uri: str) -> AbstractResourceManager:
        parsed_uri = urlparse(uri)
        if parsed_uri.scheme in {"", "json", "yaml", "yml"}:
            return JSONResourceManager(parsed_uri.path)
        else:
            raise NotImplementedError(f"Scheme \"{parsed_uri.scheme}\" is not supported.")


def _test():
    client = ResourceManager("config_test.yml")
    print(client.get("qwen_2.5"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_test())
