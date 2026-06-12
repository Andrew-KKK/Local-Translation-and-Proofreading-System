"""Local novel translation assistant."""

import ctypes
import os
import sys


def _preload_windows_runtime() -> None:
    """Avoid stale MSVC DLLs beside an Anaconda base interpreter."""
    if sys.platform != "win32" or "anaconda" not in sys.base_prefix.lower():
        return
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for dll_name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"):
        ctypes.WinDLL(os.path.join(system32, dll_name))


_preload_windows_runtime()

__version__ = "0.1.0"
