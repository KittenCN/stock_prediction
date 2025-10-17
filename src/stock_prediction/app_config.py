"""
Configuration loader with optional .env and YAML support, backed by pydantic validation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import yaml
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False
from pydantic import BaseModel, Field, ConfigDict, field_validator


# Load environment variables as early as possible so defaults stay in sync
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)


class SlidingWindowConfig(BaseModel):
    """Sliding window aggregation configuration used during feature engineering."""

    size: int = Field(5, gt=0, description="Window size for rolling aggregation")
    stride: int = Field(1, gt=0, description="Step between consecutive windows")
    agg: str = Field("mean", description="Aggregation name: mean/std/max/min/median/sum")

    @field_validator("agg")
    @classmethod
    def validate_agg(cls, value: str) -> str:
        value_lower = value.lower()
        if value_lower not in {"mean", "std", "max", "min", "median", "sum"}:
            raise ValueError("agg must be one of mean,std,max,min,median,sum")
        return value_lower


class ExternalFeatureConfig(BaseModel):
    """Specification for external (macro/industry/sentiment) feature sources."""

    name: str = Field(..., description="Display name of the external signal")
    path: str = Field(..., description="CSV file path containing external features")
    join_on: str = Field("trade_date", description="Date column used for alignment")
    ts_code_column: Optional[str] = Field(None, description="Column for stock identifier, optional")
    forward_fill: bool = Field(True, description="Forward fill missing external values")
    weight: float = Field(1.0, description="Weight applied when combining overlapping sources")
    domain: Optional[str] = Field(None, description="Domain tag: macro/industry/sentiment/other")


class FeatureSettings(BaseModel):
    """Feature engineering configuration for PTFT+VSSM and other models."""

    target_mode: str = Field("log_return", description="Target transformation: log_return|pct_return|difference|hybrid")
    return_kind: str = Field("log", description="Return calculation type: log|simple")
    return_lag: int = Field(1, ge=1, description="Lag (in days) used when computing returns")
    difference_order: int = Field(1, ge=0, description="Difference order for trend removal")
    price_columns: List[str] = Field(default_factory=lambda: ["close"], description="Columns to treat as prices")
    difference_columns: List[str] = Field(default_factory=lambda: ["close"], description="Columns to difference")
    volatility_columns: List[str] = Field(default_factory=lambda: ["pct_change", "change", "vol"], description="Columns summarised in sliding windows")
    external_sources: List[ExternalFeatureConfig] = Field(default_factory=list, description="External (macro/industry/sentiment) features")
    sliding_windows: List[SlidingWindowConfig] = Field(default_factory=list, description="Rolling window aggregations")
    multi_stock: bool = Field(True, description="Allow multi-stock joint training")
    window_ensemble_ops: List[str] = Field(default_factory=lambda: ["mean", "std"], description="Ops applied when stacking sliding windows")
    align_holiday: bool = Field(True, description="Align external features across holidays by forward filling")
    enable_direction_label: bool = Field(True, description="Generate auxiliary direction/regime labels from targets")
    enable_symbol_normalization: bool = Field(False, description="Enable per-symbol mean/std normalization with stats retention")
    use_symbol_embedding: bool = Field(False, description="Embed ts_code into learnable vector for downstream models")
    symbol_embedding_dim: int = Field(16, ge=1, description="Dimension of symbol embedding when enabled")

    @field_validator("target_mode")
    @classmethod
    def validate_target_mode(cls, value: str) -> str:
        allowed = {"log_return", "pct_return", "difference", "hybrid"}
        value_lower = value.lower()
        if value_lower not in allowed:
            raise ValueError(f"target_mode must be one of {', '.join(sorted(allowed))}")
        return value_lower

    @field_validator("return_kind")
    @classmethod
    def validate_return_kind(cls, value: str) -> str:
        value_lower = value.lower()
        if value_lower not in {"log", "simple"}:
            raise ValueError("return_kind must be 'log' or 'simple'")
        return value_lower

    @field_validator("window_ensemble_ops", mode="before")
    @classmethod
    def validate_window_ops(cls, value):
        allowed = {"mean", "std", "max", "min", "median"}
        if value is None:
            return value
        if isinstance(value, (list, tuple)):
            validated = []
            for item in value:
                item_lower = str(item).lower()
                if item_lower not in allowed:
                    raise ValueError(f"window_ensemble_ops items must be in {allowed}")
                validated.append(item_lower)
            return validated
        item_lower = str(value).lower()
        if item_lower not in allowed:
            raise ValueError(f"window_ensemble_ops items must be in {allowed}")
        return [item_lower]


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
    features: FeatureSettings = Field(default_factory=FeatureSettings, description="Feature engineering configuration")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @field_validator("batch_size", "epoch")
    @classmethod
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

        if "training" in config_dict:
            # Flatten optional nested training section for backwards compatibility
            training_cfg = config_dict.pop("training") or {}
            config_dict.update(training_cfg)

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
