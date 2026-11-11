"""Opt-in, lazy SGLang import hook used by the installer-managed .pth file.

Nothing in this module imports torch or SGLang. Each spawned Python worker
registers its own hook, then patches the scheduler only after it is loaded.
"""

import importlib
import importlib.abc
import importlib.machinery
import os
import sys


_TARGET = "sglang.srt.managers.scheduler"
_finder = None
_installed = False


def _patch(module):
    global _installed
    if _installed:
        return
    try:
        importlib.import_module("pp_chunk_trace.patch").install(module)
    except Exception as exc:
        raise RuntimeError(
            "PP chunk tracing could not be enabled for this SGLang scheduler. "
            "No valid trace will be produced. Check the patch compatibility "
            "and configuration, or unset PP_CHUNK_TRACE_ENABLE before restarting."
        ) from exc
    _installed = True
    deactivate()


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        _patch(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot locate SGLang scheduler for PP chunk tracing")
        if not hasattr(spec.loader, "exec_module"):
            raise ImportError("Unsupported SGLang scheduler loader for PP chunk tracing")
        spec.loader = _Loader(spec.loader)
        return spec


def activate():
    """Install a lazy import hook only when explicitly enabled; idempotent."""
    global _finder
    if os.environ.get("PP_CHUNK_TRACE_ENABLE") != "1":
        return False
    if _installed or _finder is not None:
        return True
    existing = sys.modules.get(_TARGET)
    if existing is not None:
        if getattr(getattr(existing, "__spec__", None), "_initializing", False):
            raise RuntimeError("Cannot enable PP tracing during scheduler import")
        _patch(existing)
    else:
        _finder = _Finder()
        sys.meta_path.insert(0, _finder)
    return True


def deactivate():
    """Remove the pending import hook; does not remove already-active wrappers."""
    global _finder
    if _finder is not None:
        if _finder in sys.meta_path:
            sys.meta_path.remove(_finder)
        _finder = None
