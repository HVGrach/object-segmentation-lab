"""Training loop with checkpointing, evaluation, and results logging."""

from __future__ import annotations

import csv
import json
import signal
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .model import OneShotModel


# ======================================================================
# Loss
# ======================================================================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits, target):
        pred = torch.sigmoid(logits)
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(self, logits, target):
        return self.bce_weight * self.bce(logits, target) + self.dice_weight * self.dice_loss(logits, target)


# ======================================================================
# Metrics
# ======================================================================
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """Compute Dice, IoU, pixel accuracy from logits and binary targets."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        smooth = 1e-6
        intersection = (preds * targets).sum()
        union_dice = preds.sum() + targets.sum()
        union_iou = preds.sum() + targets.sum() - intersection

        dice = (2.0 * intersection + smooth) / (union_dice + smooth)
        iou = (intersection + smooth) / (union_iou + smooth)
        acc = (preds == targets).float().mean()
    return {"dice": dice.item(), "iou": iou.item(), "pixel_acc": acc.item()}


def compute_iou_per_sample(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> list[float]:
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        smooth = 1e-6
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
    return [float(x.item()) for x in iou]


# ======================================================================
# Trainer
# ======================================================================
class Trainer:
    """Training orchestrator with:
    - MPS/CUDA/CPU support
    - Checkpointing (best + every N epochs + latest + on-error)
    - Gradient clipping
    - Graceful interrupt handling (Ctrl+C saves checkpoint)
    - Results table auto-append
    """

    def __init__(self, model: OneShotModel, cfg: Config, device: torch.device):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.hard_example_ids = cfg.load_hard_example_ids()

        # Directories
        self.save_dir = cfg.save_dir
        self.ckpt_dir = self.save_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        cfg.to_yaml(self.save_dir / "config.yaml")

        # Loss
        self.criterion = BCEDiceLoss(cfg.bce_weight, cfg.dice_weight).to(device)

        # Optimizer & scheduler (only trainable params)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)

        if cfg.scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5
            )

        # History
        self.history: list[dict] = []
        self.best_iou = 0.0
        self.start_epoch = 0

        # Interrupt flag
        self._interrupted = False
        self._prev_handler = None

    def _sync_device(self):
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()

    def _iter_loader_once(self, loader: DataLoader) -> dict:
        iterator = iter(loader)
        try:
            return next(iterator)
        except StopIteration as exc:
            raise RuntimeError("Dataloader is empty during preflight.") from exc

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, metrics: dict, tag: str = ""):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_iou": self.best_iou,
            "metrics": metrics,
            "config": self.cfg.to_dict(),
        }
        name = f"epoch_{epoch:03d}.pt" if not tag else f"{tag}.pt"
        torch.save(state, self.ckpt_dir / name)

    def _save_history(self):
        with open(self.save_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    def resume_from(self, checkpoint_path: str | Path):
        """Resume training from a checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_iou = ckpt.get("best_iou", 0.0)
        self.start_epoch = ckpt["epoch"] + 1
        print(f"[trainer] resumed from epoch {ckpt['epoch']}, best_iou={self.best_iou:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _move_batch(self, batch: dict) -> dict:
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device, non_blocking=True)
            else:
                moved[k] = v
        return moved

    def run_preflight(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Run one train-style and one val-style step before long training."""
        summary = {
            "device": str(self.device),
            "train_batch_ok": False,
            "val_batch_ok": False,
            "train_loss": None,
            "val_loss": None,
            "val_iou": None,
        }

        self.model.train()
        if self.model.backbone is not None:
            self.model.backbone.eval()

        train_batch = self._move_batch(self._iter_loader_once(train_loader))
        self.optimizer.zero_grad(set_to_none=True)
        train_logits = self.model(train_batch)
        train_loss = self.criterion(train_logits, train_batch["query_mask"])
        train_loss.backward()
        if self.cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.grad_clip,
            )
        self.optimizer.zero_grad(set_to_none=True)
        self._sync_device()
        summary["train_batch_ok"] = True
        summary["train_loss"] = float(train_loss.item())

        self.model.eval()
        with torch.no_grad():
            val_batch = self._move_batch(self._iter_loader_once(val_loader))
            val_logits = self.model(val_batch)
            val_loss = self.criterion(val_logits, val_batch["query_mask"])
            metrics = compute_metrics(val_logits, val_batch["query_mask"])
        self._sync_device()
        summary["val_batch_ok"] = True
        summary["val_loss"] = float(val_loss.item())
        summary["val_iou"] = float(metrics["iou"])

        self.cfg.preflight_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cfg.preflight_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(
            "[trainer] preflight OK | "
            f"train_loss={summary['train_loss']:.4f} | "
            f"val_loss={summary['val_loss']:.4f} | "
            f"val_iou={summary['val_iou']:.4f}"
        )
        return summary

    def _train_one_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        # Keep backbone frozen even in train mode
        if self.model.backbone is not None:
            self.model.backbone.eval()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._move_batch(batch)
            logits = self.model(batch)
            loss = self.criterion(logits, batch["query_mask"])

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.grad_clip,
                )
            self.optimizer.step()

            self._sync_device()

            metrics = compute_metrics(logits.detach(), batch["query_mask"])
            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            n_batches += 1

            if self._interrupted:
                break

        n = max(n_batches, 1)
        return {"train_loss": total_loss / n, "train_dice": total_dice / n, "train_iou": total_iou / n}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        n_batches = 0
        hard_ious: list[float] = []
        easy_ious: list[float] = []

        for batch in loader:
            query_ids = batch.get("query_id")
            batch = self._move_batch(batch)
            logits = self.model(batch)
            loss = self.criterion(logits, batch["query_mask"])

            self._sync_device()

            metrics = compute_metrics(logits, batch["query_mask"])
            total_loss += loss.item()
            total_dice += metrics["dice"]
            total_iou += metrics["iou"]
            n_batches += 1

            if query_ids is not None and self.hard_example_ids:
                sample_ious = compute_iou_per_sample(logits, batch["query_mask"])
                for query_id, sample_iou in zip(query_ids, sample_ious):
                    if query_id in self.hard_example_ids:
                        hard_ious.append(sample_iou)
                    else:
                        easy_ious.append(sample_iou)

        n = max(n_batches, 1)
        payload = {"val_loss": total_loss / n, "val_dice": total_dice / n, "val_iou": total_iou / n}
        if hard_ious:
            payload["hard_val_iou"] = sum(hard_ious) / len(hard_ious)
        if easy_ious:
            payload["easy_val_iou"] = sum(easy_ious) / len(easy_ious)
        return payload

    # ------------------------------------------------------------------
    def _handle_interrupt(self, signum, frame):
        print("\n[trainer] Ctrl+C detected — saving emergency checkpoint ...")
        self._interrupted = True

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop with all safety features."""
        # Set up Ctrl+C handler
        self._prev_handler = signal.signal(signal.SIGINT, self._handle_interrupt)

        try:
            if self.cfg.run_preflight:
                self.run_preflight(train_loader, val_loader)
            self._train_loop(train_loader, val_loader)
        except Exception as e:
            print(f"\n[trainer] ERROR: {e}")
            traceback.print_exc()
            self._save_checkpoint(
                epoch=self.start_epoch + len(self.history),
                metrics=self.history[-1] if self.history else {},
                tag="emergency",
            )
            print(f"[trainer] emergency checkpoint saved to {self.ckpt_dir}/emergency.pt")
            raise
        finally:
            signal.signal(signal.SIGINT, self._prev_handler or signal.SIG_DFL)

    def _train_loop(self, train_loader: DataLoader, val_loader: DataLoader):
        print(f"\n{'='*60}")
        print(f"  Experiment: {self.cfg.experiment_name}")
        print(f"  Backbone:   DINOv2-{self.cfg.backbone_size.upper()} (frozen)")
        print(f"  Fusion:     {self.cfg.fusion_type}")
        print(f"  Epochs:     {self.cfg.epochs}")
        print(f"  Device:     {self.device}")
        print(f"  Save dir:   {self.save_dir}")
        print(f"{'='*60}\n")

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} total\n")

        for epoch in range(self.start_epoch, self.cfg.epochs):
            t0 = time.time()
            train_metrics = self._train_one_epoch(train_loader)

            if self._interrupted:
                self._save_checkpoint(epoch, train_metrics, tag="interrupted")
                self._save_history()
                print(f"[trainer] interrupted at epoch {epoch}, checkpoint saved.")
                return

            val_metrics = self._validate(val_loader)
            elapsed = time.time() - t0

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["val_iou"])
            else:
                self.scheduler.step()

            # Merge metrics
            row = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_sec": round(elapsed, 1),
            }
            self.history.append(row)

            # Log
            print(
                f"  Epoch {epoch:3d}/{self.cfg.epochs} | "
                f"train_loss={row['train_loss']:.4f} dice={row['train_dice']:.4f} | "
                f"val_loss={row['val_loss']:.4f} dice={row['val_dice']:.4f} iou={row['val_iou']:.4f}"
                + (f" hard_iou={row['hard_val_iou']:.4f}" if "hard_val_iou" in row else "")
                + (f" easy_iou={row['easy_val_iou']:.4f}" if "easy_val_iou" in row else "")
                + " | "
                f"{elapsed:.1f}s"
            )

            # Checkpoints
            if val_metrics["val_iou"] > self.best_iou:
                self.best_iou = val_metrics["val_iou"]
                self._save_checkpoint(epoch, row, tag="best")
                print(f"  >>> new best iou: {self.best_iou:.4f}")

            if (epoch + 1) % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(epoch, row)

            self._save_checkpoint(epoch, row, tag="latest")
            self._save_history()

        # Final summary
        print(f"\n{'='*60}")
        print(f"  Training complete. Best val IoU: {self.best_iou:.4f}")
        print(f"  Checkpoints: {self.ckpt_dir}")
        print(f"{'='*60}\n")

        # Append to results table
        self._update_results_table()

    # ------------------------------------------------------------------
    def _update_results_table(self):
        """Append one row to the global CSV results table."""
        csv_path = self.cfg.results_table_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        best_epoch = max(self.history, key=lambda r: r["val_iou"]) if self.history else {}
        row = {
            "experiment": self.cfg.experiment_name,
            "backbone": f"dinov2_{self.cfg.backbone_size}",
            "fusion": self.cfg.fusion_type,
            "img_size": self.cfg.img_size,
            "epochs": self.cfg.epochs,
            "best_val_iou": round(best_epoch.get("val_iou", 0), 4),
            "best_val_dice": round(best_epoch.get("val_dice", 0), 4),
            "best_hard_val_iou": round(best_epoch.get("hard_val_iou", 0), 4),
            "best_easy_val_iou": round(best_epoch.get("easy_val_iou", 0), 4),
            "best_epoch": best_epoch.get("epoch", -1),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "lr": self.cfg.lr,
            "batch_size": self.cfg.batch_size,
            "use_cached_features": self.cfg.use_cached_features,
            "checkpoint_dir": str(self.ckpt_dir),
        }

        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"[trainer] results appended to {csv_path}")
