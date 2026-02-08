"""Compatibility bridge to legacy top-level package `simulator`."""

from importlib import import_module as _import_module

_legacy = _import_module("simulator")
__path__ = _legacy.__path__
__all__ = getattr(_legacy, "__all__", [])

# Re-export public names for `from prepbench.simulator import ...`
for _k, _v in _legacy.__dict__.items():
    if not _k.startswith("_"):
        globals()[_k] = _v
