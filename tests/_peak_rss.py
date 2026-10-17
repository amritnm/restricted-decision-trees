"""Peak resident memory for the memory tests' subprocess scripts.

Each script runs in a fresh interpreter (``python -c``) and measures how far a
call raises the process's peak resident memory, so it starts with PEAK_RSS, the
source of a ``peak()`` function that works on each platform.
"""

import sys

import pytest

PEAK_RSS = '''
import sys

def peak():
    """This process's peak resident memory, in bytes."""
    if sys.platform == "win32":
        import psutil
        return psutil.Process().memory_info().peak_wset  # peak working set
    if sys.platform.startswith("linux"):
        # VmHWM: this process's own peak. Linux carries ru_maxrss across execve, so it would
        # start at the RSS of the (pytest) process that forked this one.
        with open("/proc/self/status") as f:
            return next(int(line.split()[1]) for line in f if line.startswith("VmHWM:")) * 1024
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024
'''


def require_peak_rss():
    """Skip the calling test, saying why, where peak() cannot measure."""
    if sys.platform == "win32":
        pytest.importorskip("psutil", reason="the peak working set on Windows is read with psutil")
    else:
        pytest.importorskip("resource")
