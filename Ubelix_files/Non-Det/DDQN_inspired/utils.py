def config_to_dict(cfg):
    """Serialize a dataclass (e.g. Config) into plain python types"""
    out = {}
    for k, v in cfg.__dict__.items():
        if hasattr(v, '__dict__'):      # recurse if nested dataclass
            out[k] = config_to_dict(v)
        else:
            out[k] = v
    return out