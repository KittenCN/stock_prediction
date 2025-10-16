"""
Configuration loader with optional .env and YAML support, backed by pydantic validation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False
from pydantic import BaseModel, Field, validator


# Load environment variables as early as possible so defaults stay in sync
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


class AppConfig(BaseModel):
    """Application level settings used by training and inference utilities."""

    train_pkl_path: str = Field(..., description="Path to the training data queue (pkl)")
    png_path: str = Field(..., description="Directory for generated charts")
    model_path: str = Field(..., description="Directory for persisted model checkpoints")
    batch_size: int = Field(32, description="Mini-batch size")
    epoch: int = Field(2, description="Number of training epochs")
    api: str = Field("akshare", description="Upstream data API source")
    # Training related
    scheduler_type: str = Field("none", description="LR scheduler type: none|step|plateau")
    scheduler_step_size: int = Field(10, description="StepLR step_size")
    scheduler_gamma: float = Field(0.1, description="LR decay factor")
    early_stopping_patience: int = Field(5, description="EarlyStopping patience")
    early_stopping_min_delta: float = Field(1e-4, description="EarlyStopping min improvement")
    seed: int = Field(42, description="Global random seed")

    @validator("batch_size", "epoch")
    def validate_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be a positive integer")
        return value

    @classmethod
    def from_env_and_yaml(cls, yaml_path: Optional[str] = None) -> "AppConfig":
        """Load configuration from an optional YAML file and override with environment variables."""

        config_dict: dict[str, object] = {}
        if yaml_path and Path(yaml_path).exists():
            with open(yaml_path, "r", encoding="utf-8") as fh:
                config_dict = yaml.safe_load(fh) or {}

        env_map = {
            "train_pkl_path": os.getenv("TRAIN_PKL_PATH"),
            "png_path": os.getenv("PNG_PATH"),
            "model_path": os.getenv("MODEL_PATH"),
            "batch_size": os.getenv("BATCH_SIZE"),
            "epoch": os.getenv("EPOCH"),
            "api": os.getenv("API"),
            "scheduler_type": os.getenv("SCHEDULER_TYPE"),
            "scheduler_step_size": os.getenv("SCHEDULER_STEP_SIZE"),
            "scheduler_gamma": os.getenv("SCHEDULER_GAMMA"),
            "early_stopping_patience": os.getenv("EARLY_STOPPING_PATIENCE"),
            "early_stopping_min_delta": os.getenv("EARLY_STOPPING_MIN_DELTA"),
            "seed": os.getenv("SEED"),
        }
        for key, value in env_map.items():
            if value is None:
                continue
            if key in {"batch_size", "epoch", "scheduler_step_size", "early_stopping_patience", "seed"}:
                config_dict[key] = int(value)
            elif key in {"scheduler_gamma", "early_stopping_min_delta"}:
                config_dict[key] = float(value)
            else:
                config_dict[key] = value

        return cls(**config_dict)

    def get_model_path(self, model_type: str, symbol: str = "GenericData") -> str:
        """Keep compatibility with config.py by mirroring the model path layout."""

        symbol_clean = symbol.replace(".", "")
        model_dir = Path(self.model_path) / symbol_clean / model_type.upper()
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir / model_type.upper())


# Usage example:
# config = AppConfig.from_env_and_yaml("config/config.yaml")
# print(config)
