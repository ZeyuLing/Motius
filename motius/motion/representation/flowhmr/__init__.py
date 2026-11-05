__all__ = ["FlowHMRMotionCodec"]


def __getattr__(name):
    # Body models import the geometry submodule. Loading the codec eagerly here
    # would import those same body models before their initialization finishes.
    if name == "FlowHMRMotionCodec":
        from .codec import FlowHMRMotionCodec
        return FlowHMRMotionCodec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
