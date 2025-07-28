#!/usr/bin/env python3

__author__ = "xi"
__all__ = [
    "Timer",
    "TimedLRUCache",
]

import threading
import time
from collections import OrderedDict
from functools import wraps


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


class TimedLRUCache:

    def __init__(self, max_size: int = 256, timeout: float = 12):
        self.maxsize = max_size
        self.timeout = timeout  # seconds

        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            now = time.time()

            with self.lock:
                # clean expired items
                if key in self.cache:
                    result, timestamp = self.cache[key]
                    if now - timestamp < self.timeout:
                        self.cache.move_to_end(key)
                        return result
                    else:
                        del self.cache[key]  # expired

                # compute new result
                result = func(*args, **kwargs)
                self.cache[key] = (result, now)
                self.cache.move_to_end(key)

                # exceed the max size
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)  # remove the oldest item

                return result

        return wrapped
