import os
import re
import sys
import logging
from io import StringIO

import wgpu
from wgpu.utils import get_default_device  # noqa: F401 - imported by tests
import pytest


class LogCaptureHandler(logging.StreamHandler):
    _ANSI_ESCAPE_SEQ = re.compile(r"\x1b\[[\d;]+m")

    def __init__(self):
        super().__init__(StringIO())
        self.records = []

    def emit(self, record):
        record.msg = self._ANSI_ESCAPE_SEQ.sub("", record.msg)
        self.records.append(record)
        super().emit(record)

    def reset(self):
        self.records = []
        self.stream = StringIO()

    @property
    def text(self):
        f = logging.Formatter()
        return "\n".join(f.format(r) for r in self.records)


def run_tests(scope):
    """Run all test functions in the given scope."""
    caplog = LogCaptureHandler()
    for func in list(scope.values()):
        if callable(func) and func.__name__.startswith("test_"):
            nargs = func.__code__.co_argcount
            argnames = [func.__code__.co_varnames[i] for i in range(nargs)]
            try:
                if not argnames:
                    print(f"Running {func.__name__} ...")
                    func()
                elif argnames == ["caplog"]:
                    print(f"Running {func.__name__} ...")
                    logging.root.addHandler(caplog)
                    caplog.reset()
                    func(caplog)
                    logging.root.removeHandler(caplog)
                else:
                    print(f"SKIPPING {func.__name__} because it needs args")
            except pytest.skip.Exception:
                print(f"SKIPPING {func.__name__} by pytest skip")
    print("Done")


def iters_equal(iter1, iter2):
    iter1, iter2 = list(iter1), list(iter2)
    if len(iter1) == len(iter2):
        if all(iter1[i] == iter2[i] for i in range(len(iter1))):
            return True
    return False


def get_default_adapter_summary():
    """Get description of adapter, or None when no adapter is available."""
    try:
        adapter = wgpu.gpu.request_adapter_sync()
    except RuntimeError:
        return None  # lib not available, or no adapter on this system
    return adapter.summary


adapter_summary = get_default_adapter_summary()
can_use_wgpu_lib = bool(adapter_summary)

is_ci = bool(os.getenv("CI", None))
is_pypy = sys.implementation.name == "pypy"
