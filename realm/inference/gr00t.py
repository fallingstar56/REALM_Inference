from realm.inference.gr00t_n17 import Gr00tN17Client, compute_gr00t_n17_eef_9d


class Gr00tClient(Gr00tN17Client):
    """Legacy compatibility alias. Prefer Gr00tN17Client for GR00T N1.7."""


def compute_eef_9d(cartesian_position):
    return compute_gr00t_n17_eef_9d(cartesian_position)
