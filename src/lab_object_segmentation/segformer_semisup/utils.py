from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def infer_camera_group(image_path: str | Path) -> str:
    path = Path(image_path)
    prefix = path.name.split("_", 1)[0]
    parts = prefix.split(".")
    if len(parts) == 4 and all(part.isdigit() for part in parts):
        return prefix
    return path.parent.name or prefix


def split_labeled_samples_grouped(
    samples: list[tuple[Path, Path]],
    val_split: float,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for sample in samples:
        group = infer_camera_group(sample[0])
        grouped.setdefault(group, []).append(sample)

    group_names = list(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_names)

    target_val_size = max(1, int(round(len(samples) * val_split)))
    val_groups: list[str] = []
    val_size = 0
    for group_name in group_names:
        if val_size >= target_val_size and val_groups:
            break
        val_groups.append(group_name)
        val_size += len(grouped[group_name])

    val_group_set = set(val_groups)
    train_samples = [sample for group, items in grouped.items() if group not in val_group_set for sample in items]
    val_samples = [sample for group, items in grouped.items() if group in val_group_set for sample in items]

    if not train_samples or not val_samples:
        raise RuntimeError("Grouped split failed to produce both train and validation samples.")

    return train_samples, val_samples


def build_pseudo_stats(
    pseudo_mask,
    confidence,
    uncertainty,
    confidence_threshold: float,
    uncertainty_threshold: float,
) -> dict:
    reliable = (
        (confidence >= confidence_threshold)
        & (uncertainty <= uncertainty_threshold)
    ).astype("uint8")

    object_ratio = float(pseudo_mask.mean())
    reliable_ratio = float(reliable.mean())
    avg_confidence = float(confidence.mean())

    fg_mask = pseudo_mask.astype(bool)
    bg_mask = ~fg_mask
    fg_reliable_ratio = float(reliable[fg_mask].mean()) if fg_mask.any() else 1.0
    bg_reliable_ratio = float(reliable[bg_mask].mean()) if bg_mask.any() else 1.0

    score = (
        0.50 * reliable_ratio
        + 0.35 * avg_confidence
        + 0.15 * fg_reliable_ratio
    )

    return {
        "reliable": reliable,
        "reliable_ratio": reliable_ratio,
        "object_ratio": object_ratio,
        "avg_confidence": avg_confidence,
        "fg_reliable_ratio": fg_reliable_ratio,
        "bg_reliable_ratio": bg_reliable_ratio,
        "selection_score": score,
    }


def should_accept_pseudo_sample(
    stats: dict,
    *,
    min_reliable_ratio: float,
    min_mean_confidence: float,
    min_fg_reliable_ratio: float,
    min_object_ratio: float,
    max_object_ratio: float,
) -> bool:
    return bool(
        stats["reliable_ratio"] >= min_reliable_ratio
        and stats["avg_confidence"] >= min_mean_confidence
        and stats["fg_reliable_ratio"] >= min_fg_reliable_ratio
        and min_object_ratio <= stats["object_ratio"] <= max_object_ratio
    )


def boundary_loss_with_reliability(
    pred_boundary: torch.Tensor,
    target_boundary: torch.Tensor,
    reliability_mask: torch.Tensor | None,
    pos_weight: torch.Tensor | float,
) -> torch.Tensor:
    target_boundary = target_boundary.float()
    if not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(float(pos_weight), device=pred_boundary.device)
    else:
        pos_weight = pos_weight.to(device=pred_boundary.device, dtype=pred_boundary.dtype)

    loss_map = F.binary_cross_entropy_with_logits(
        pred_boundary,
        target_boundary,
        pos_weight=pos_weight,
        reduction="none",
    )
    if reliability_mask is None:
        return loss_map.mean()

    reliability = reliability_mask.float()
    if reliability.ndim == 3:
        reliability = reliability.unsqueeze(1)
    return (loss_map * reliability).sum() / reliability.sum().clamp_min(1.0)


def binary_dice_score(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    ignore_index: int = 255,
    eps: float = 1e-7,
) -> float:
    valid = target_mask != ignore_index
    if valid.sum() == 0:
        return 0.0

    pred = pred_mask[valid].astype(bool)
    target = target_mask[valid].astype(bool)
    intersection = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    return float((2.0 * intersection + eps) / (denom + eps))


def tune_binary_threshold(
    probabilities: list[np.ndarray],
    targets: list[np.ndarray],
    *,
    threshold_grid: list[float],
    ignore_index: int = 255,
) -> dict:
    if not probabilities or not targets:
        default_threshold = float(threshold_grid[0]) if threshold_grid else 0.5
        return {
            "best_threshold": default_threshold,
            "best_dice": 0.0,
            "rows": [],
        }

    rows: list[dict] = []
    best_threshold = float(threshold_grid[0]) if threshold_grid else 0.5
    best_dice = -1.0

    for threshold in threshold_grid:
        threshold = float(threshold)
        dices = [
            binary_dice_score(
                pred_mask=(prob >= threshold).astype(np.uint8),
                target_mask=target,
                ignore_index=ignore_index,
            )
            for prob, target in zip(probabilities, targets)
        ]
        mean_dice = float(np.mean(dices)) if dices else 0.0
        row = {
            "threshold": threshold,
            "dice": mean_dice,
        }
        rows.append(row)
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = threshold

    return {
        "best_threshold": float(best_threshold),
        "best_dice": float(best_dice),
        "rows": rows,
    }
