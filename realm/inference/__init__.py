__all__ = ["InferenceClient", "Gr00tClient", "extract_from_obs"]


def __getattr__(name):
    if name == "InferenceClient":
        from realm.inference.client import InferenceClient

        return InferenceClient
    if name == "Gr00tClient":
        from realm.inference.gr00t import Gr00tClient

        return Gr00tClient
    if name == "extract_from_obs":
        from realm.inference.utils import extract_from_obs

        return extract_from_obs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
