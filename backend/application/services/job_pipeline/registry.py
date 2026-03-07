from __future__ import annotations

import importlib
import inspect
from typing import Dict

from domain.interfaces import JobRunner


class JobRunnerRegistry:
    """Config-driven runner resolver (runner_ref -> runnable object)."""

    def __init__(self) -> None:
        self._cache: Dict[str, JobRunner] = {}

    def resolve(self, runner_ref: str) -> JobRunner:
        ref = str(runner_ref or "").strip()
        if not ref:
            raise ValueError("runner_ref is required")
        cached = self._cache.get(ref)
        if cached is not None:
            return cached

        if ":" in ref:
            module_name, symbol_name = ref.split(":", 1)
        else:
            module_name, symbol_name = ref.rsplit(".", 1)
        module = importlib.import_module(module_name)
        symbol = getattr(module, symbol_name)
        instance = symbol() if inspect.isclass(symbol) else symbol
        if not hasattr(instance, "run"):
            raise TypeError(f"Resolved runner '{ref}' does not implement run(context)")
        self._cache[ref] = instance
        return instance
