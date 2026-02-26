"""Shim module to proxy the standard library 'types' module.

This avoids shadowing issues when running scripts from the package directory.
"""

from __future__ import annotations

import importlib.util
import os
import sysconfig


def _load_stdlib_types():
    stdlib_path = sysconfig.get_paths().get("stdlib")
    if not stdlib_path:
        return None
    types_path = os.path.join(stdlib_path, "types.py")
    spec = importlib.util.spec_from_file_location("types", types_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_stdlib_types = _load_stdlib_types()
if _stdlib_types is not None:
    globals().update(_stdlib_types.__dict__)
else:
    raise ImportError("Failed to load stdlib 'types' module")
