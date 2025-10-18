"""
Reusable Trainer with LR scheduler & EarlyStopping.
- Handles train/validate loops
- Integrates torch.optim.lr_scheduler
- Supports early stopping on validation loss
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .init import TQDM_NCOLS


@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    min_delta: float = 1e-4
    mode: str = "min"  # "min" for loss, "max" for score


class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig):
        self.cfg = cfg
        self.best: Optional[float] = None
        self.num_bad: int = 0
        self.should_stop: bool = False

    def step(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return False
        improved = (current < self.best - self.cfg.min_delta) if self.cfg.mode == "min" else (current > self.best + self.cfg.min_delta)
        if improved:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.cfg.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: torch.device,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        early_stopping: Optional[EarlyStopping] = None,
        epoch_count: int = 1,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        use_amp: bool = False,
        callbacks: Optional[Dict[str, Callable[[Dict[str, Any]], None]]] = None,
        show_progress: bool = False,
        progress_bar_ncols: int = TQDM_NCOLS,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.epoch_count = epoch_count
        self.scaler = scaler
        self.use_amp = use_amp
        self.callbacks = callbacks or {}
        self.show_progress = show_progress
        self.progress_bar_ncols = progress_bar_ncols

    def _run_callbacks(self, hook: str, context: Dict[str, Any]) -> None:
        cb = self.callbacks.get(hook)
        if cb:
            cb(context)

    def train_one_epoch(self, batch_loss_list=None) -> float:
        self.model.train()
        total_loss = 0.0
        batch_bar: Optional[tqdm] = None
        if self.show_progress:
            try:
                total_batches = len(self.train_loader)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                total_batches = None
            if total_batches and total_batches > 0:
                batch_bar = tqdm(
                    total=total_batches,
                    leave=False,
                    ncols=self.progress_bar_ncols,
                    desc="train-batch",
                )
        for batch_idx, batch in enumerate(self.train_loader):
            # Null check, compatible with legacy behavior
            if batch is None:
                if batch_bar:
                    batch_bar.update(1)
                continue
            inputs = targets = symbol_index = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    inputs, targets, symbol_index = batch
                elif len(batch) == 2:
                    inputs, targets = batch
                else:
                    if len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                        if len(batch) >= 3:
                            symbol_index = batch[2]
                    else:
                        if batch_bar:
                            batch_bar.update(1)
                        continue
            else:
                # fallback: single tensor as inputs, targets must be provided by dataset
                if batch is None or len(batch) < 2 or batch[0] is None or batch[1] is None:
                    if batch_bar:
                        batch_bar.update(1)
                    continue
                inputs, targets = batch[0], batch[1]

            if inputs is None or targets is None:
                if batch_bar:
                    batch_bar.update(1)
                continue

            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if symbol_index is not None:
                symbol_index = symbol_index.to(self.device, non_blocking=True).long()

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp and self.scaler is not None:
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=True):
                    outputs = self._forward_model(inputs, symbol_index)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self._forward_model(inputs, symbol_index)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            loss_value = loss.detach().item()
            total_loss += loss_value
            if batch_loss_list is not None:
                batch_loss_list.append(loss_value)
            if batch_bar:
                batch_bar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "avg": f"{total_loss / (batch_idx + 1):.4f}",
                })
                batch_bar.update(1)
        if batch_bar:
            batch_bar.close()
        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        for batch in self.val_loader:
            inputs = targets = symbol_index = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    inputs, targets, symbol_index = batch
                elif len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    if len(batch) >= 3:
                        symbol_index = batch[2]
            else:
                inputs, targets = batch[0], batch[1]
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if symbol_index is not None:
                symbol_index = symbol_index.to(self.device, non_blocking=True).long()
            outputs = self._forward_model(inputs, symbol_index)
            loss = self.criterion(outputs, targets)
            total_loss += loss.detach().item()
        return total_loss / max(1, len(self.val_loader))

    def _forward_model(self, inputs: torch.Tensor | Tuple[torch.Tensor, ...], symbol_index: Optional[torch.Tensor]) -> torch.Tensor:
        """Unified forward helper supporting optional symbol embeddings."""
        if isinstance(inputs, (tuple, list)):
            if symbol_index is not None:
                return self.model(*inputs, symbol_index=symbol_index)
            return self.model(*inputs)
        if symbol_index is not None:
            try:
                return self.model(inputs, symbol_index=symbol_index)
            except TypeError:
                return self.model(inputs)
        return self.model(inputs)

    def fit(self) -> Dict[str, Any]:
        history = {"train_loss": [], "val_loss": []}
        all_batch_loss = []
        for epoch in range(self.epoch_count):
            self._run_callbacks("on_epoch_begin", {"epoch": epoch})
            batch_loss_list = []
            train_loss = self.train_one_epoch(batch_loss_list)
            val_loss = self.validate()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            all_batch_loss.extend(batch_loss_list)

            # Scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if hasattr(self.scheduler, "__class__") and self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            # Early stopping
            if self.early_stopping is not None and not (val_loss != val_loss):  # check NaN
                prev_best = self.early_stopping.best
                stop = self.early_stopping.step(val_loss)
                # Trigger on_improve when best changes
                if self.early_stopping.best is not None and self.early_stopping.best != prev_best:
                    self._run_callbacks("on_improve", {"epoch": epoch, "best": self.early_stopping.best, "train_loss": train_loss, "val_loss": val_loss})
                if stop:
                    self._run_callbacks("on_early_stop", {"epoch": epoch, "best": self.early_stopping.best})
                    break

            self._run_callbacks("on_epoch_end", {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        history["batch_loss"] = all_batch_loss
        return history
