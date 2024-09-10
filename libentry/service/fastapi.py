#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "run_service",
]

import uvicorn
from fastapi import FastAPI

from libentry.api import list_api_info


def run_service(service, host: str, port: int):
    api_info_list = list_api_info(service)
    if len(api_info_list) == 0:
        print("No API found, nothing to serve.")
        return

    app = FastAPI()
    for fn, api_info in api_info_list:
        method = api_info.method
        path = api_info.path
        print(method, path)

        if method == "GET":
            app.get(path)(fn)
        elif method == "POST":
            app.post(path)(fn)
        else:
            raise RuntimeError(f"Unsupported method \"{method}\" for ")
    uvicorn.run(app, host=host, port=port)
