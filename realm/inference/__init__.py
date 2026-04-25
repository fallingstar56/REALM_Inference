__all__ = ["InferenceClient", "Gr00tN16Client", "Gr00tN17Client", "extract_from_obs"]


def __getattr__(name):
    if name == "InferenceClient":
        from realm.inference.client import InferenceClient

        return InferenceClient
    if name == "Gr00tN16Client":
        from realm.inference.gr00t_n16 import Gr00tN16Client

        return Gr00tN16Client
    if name == "Gr00tN17Client":
        from realm.inference.gr00t_n17 import Gr00tN17Client

        return Gr00tN17Client
    if name == "Gr00tClient":
        from realm.inference.gr00t import Gr00tClient

        return Gr00tClient
    if name == "extract_from_obs":
        from realm.inference.utils import extract_from_obs

        return extract_from_obs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
