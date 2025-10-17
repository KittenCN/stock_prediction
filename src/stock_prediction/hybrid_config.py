def get_adaptive_hybrid_config(size_hint: str = "auto", data_size: int = 0) -> dict:
    """
    Adaptively adjust Hybrid model configuration based on data size or user specification
    Args:
        size_hint: Model size hint ("auto", "tiny", "small", "medium", "large", "full")
        data_size: Number of training samples (only used when size_hint="auto")
    Returns:
        dict: Configuration dictionary containing hidden_dim and branch_config
    """
    configs = {
        "tiny": {
            "hidden_dim": 32,
            "branch_config": {
                "legacy": True,
                "ptft": False,
                "vssm": False,
                "diffusion": False,
                "graph": False,
            },
            "description": "Minimal config (for < 500 samples)",
        },
        "small": {
            "hidden_dim": 64,
            "branch_config": {
                "legacy": True,
                "ptft": False,
                "vssm": False,
                "diffusion": False,
                "graph": False,
            },
            "description": "Lightweight config (for 500-1000 samples)",
        },
        "medium": {
            "hidden_dim": 128,
            "branch_config": {
                "legacy": True,
                "ptft": True,
                "vssm": False,
                "diffusion": False,
                "graph": False,
            },
            "description": "Standard config (for 1000-5000 samples)",
        },
        "large": {
            "hidden_dim": 160,
            "branch_config": {
                "legacy": True,
                "ptft": True,
                "vssm": True,
                "diffusion": False,
                "graph": False,
            },
            "description": "Enhanced config (for 5000-10000 samples)",
        },
        "full": {
            "hidden_dim": 160,
            "branch_config": {
                "legacy": True,
                "ptft": True,
                "vssm": True,
                "diffusion": True,
                "graph": True,
            },
            "description": "Full config (for >= 10000 samples)",
        },
    }
    # Manual selection
    if size_hint != "auto" and size_hint in configs:
        return configs[size_hint]
    # Auto selection
    if data_size < 500:
        selected = "tiny"
    elif data_size < 1000:
        selected = "small"
    elif data_size < 5000:
        selected = "medium"
    elif data_size < 10000:
        selected = "large"
    else:
        selected = "full"
    return configs[selected]
