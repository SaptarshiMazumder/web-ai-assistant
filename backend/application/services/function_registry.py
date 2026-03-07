from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from domain.platform_profiles import get_function_config


FunctionHandler = Callable[[Dict[str, Any], Dict[str, Any]], Any]


class FunctionRegistry:
    """Config-driven function execution registry."""

    def __init__(self) -> None:
        self._handlers: Dict[str, FunctionHandler] = {}

    def register(self, function_id: str, handler: FunctionHandler) -> None:
        fid = str(function_id or "").strip()
        if not fid:
            raise ValueError("function_id is required")
        self._handlers[fid] = handler

    def execute(self, function_id: str, args: Dict[str, Any], context: Dict[str, Any]) -> Any:
        fid = str(function_id or "").strip()
        handler = self._handlers.get(fid)
        if handler is None:
            raise KeyError(f"Unknown function_id: {fid}")
        return handler(args or {}, context or {})

    def resolve_from_suggested_type(self, suggested_type: str) -> Optional[str]:
        cfg = get_function_config()
        mapping = cfg.get("suggested_type_to_function")
        if not isinstance(mapping, dict):
            return None
        fid = mapping.get(str(suggested_type or "").strip())
        if not fid:
            return None
        return str(fid).strip() or None
