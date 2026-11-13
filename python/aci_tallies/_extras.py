"""Optional dependencies, imported when first needed."""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str) -> ModuleType:
    """`module`, or an ImportError saying how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"this needs {module}: pip install 'aci-tallies[{extra}]' (or pip install {module})"
        ) from e
