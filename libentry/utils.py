#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "Timer"
]

import time


class Timer:

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.cost = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

    def start(self):
        self.start_time = time.perf_counter()
        self.end_time = None
        self.cost = None

    def end(self):
        if self.start_time is None:
            raise RuntimeError("The timer has not started yet.")
        self.end_time = time.perf_counter()
        self.cost = self.end_time - self.start_time
