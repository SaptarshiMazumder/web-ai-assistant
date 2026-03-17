from __future__ import annotations

from contextvars import ContextVar, Token

_RUNTIME_MODE: ContextVar[bool] = ContextVar("chat_runtime_mode", default=False)


def set_runtime_mode(enabled: bool) -> Token:
    return _RUNTIME_MODE.set(bool(enabled))


def reset_runtime_mode(token: Token | None) -> None:
    if token is None:
        return
    try:
        _RUNTIME_MODE.reset(token)
    except (ValueError, RuntimeError):
        # A stream/task boundary can execute in a different context from where
        # the token was created; in that case just force-disable runtime mode.
        _RUNTIME_MODE.set(False)


def get_runtime_mode() -> bool:
    return bool(_RUNTIME_MODE.get())
