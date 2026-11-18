__all__ = ["GALBOT_CFG"]


def __getattr__(name):
    if name == "GALBOT_CFG":
        from .galbot import GALBOT_CFG

        return GALBOT_CFG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
