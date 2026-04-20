def resolve_rollout_horizon(model_type, horizon):
    if horizon is not None:
        return horizon
    return 15 if model_type in {"gr00t_n17", "GR00T"} else 8