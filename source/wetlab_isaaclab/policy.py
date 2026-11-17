"""Algorithm-neutral policy API shared by training and evaluation."""

from __future__ import annotations

from typing import Any, Protocol


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: dict[str, Any]) -> Any: ...


class HoldPolicy:
    """A smoke-test policy that holds the environment's default pose."""

    def reset(self) -> None:
        pass

    def act(self, observation: dict[str, Any]) -> None:
        return None
