#!/usr/bin/env python3

import functools
import logging
import sys
from importlib import import_module

__all__ = [
    'LoggerFactory',
    'logger',
]

tqdm = import_module('tqdm')
if tqdm is not None:
    tqdm = tqdm.tqdm


class TqdmHandler(logging.Handler):

    # noinspection PyBroadException
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class LoggerFactory:

    @staticmethod
    def create_logger(name=None, level=logging.INFO):
        if name is None:
            raise ValueError('name for logger cannot be None')

        formatter = logging.Formatter(
            '[%(asctime)s] '
            '[%(levelname)s] '
            '[%(filename)s:%(lineno)d:%(funcName)s] '
            '%(message)s'
        )

        handler = logging.StreamHandler(stream=sys.stdout) if tqdm is None else TqdmHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        logger_.addHandler(handler)

        logger_.warning_once = functools.lru_cache(logger_.warning)
        return logger_


logger = LoggerFactory.create_logger(name='LibEntry', level=logging.INFO)
