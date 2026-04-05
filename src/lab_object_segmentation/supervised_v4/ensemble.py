from __future__ import annotations

from typing import Sequence

import numpy as np


PRESET_MEMBERS = {
    "conservative3": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold1_384_finetune_from_moderate", 1.0 / 3.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
    ],
    "blend4": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold1_384_finetune_from_moderate", 1.0 / 6.0),
        ("segformer_b2_fold1_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
    ],
    "lovasz3": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold1_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 3.0),
    ],
    "wide6": [
        ("segformer_b2_fold0_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold0_320_moderate_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold1_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold1_384_finetune_from_moderate", 1.0 / 6.0),
        ("segformer_b2_fold2_320_moderate_lovasz_ls003_ema", 1.0 / 6.0),
        ("segformer_b2_fold2_320_moderate_ls003_ema", 1.0 / 6.0),
    ],
}

AGGREGATION_CHOICES = ("weighted_mean", "mean", "hard_vote", "max_prob")


def normalize_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        raise ValueError("weights must not be empty")
    total = float(sum(float(weight) for weight in weights))
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return [float(weight) / total for weight in weights]


def resolve_member_specs(
    preset: str | None = None,
    run_names: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    if run_names is not None:
        names = [str(name) for name in run_names]
        if not names:
            raise ValueError("run_names must not be empty")
        if weights is None:
            resolved_weights = [1.0 / len(names)] * len(names)
        else:
            if len(weights) != len(names):
                raise ValueError("weights must have the same length as run_names")
            resolved_weights = normalize_weights(weights)
        return list(zip(names, resolved_weights))

    if preset is None:
        raise ValueError("preset is required when run_names are not provided")
    if preset not in PRESET_MEMBERS:
        raise ValueError(f"Unknown preset: {preset}")

    members = PRESET_MEMBERS[preset]
    return list(zip([run_name for run_name, _ in members], normalize_weights([weight for _, weight in members])))


def aggregate_probability_maps(
    probability_maps: Sequence[np.ndarray],
    weights: Sequence[float],
    aggregation: str,
    member_threshold: float = 0.5,
) -> np.ndarray:
    if aggregation not in AGGREGATION_CHOICES:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if not probability_maps:
        raise ValueError("probability_maps must not be empty")

    stack = np.stack([np.asarray(prob_map, dtype=np.float32) for prob_map in probability_maps], axis=0)
    if aggregation == "weighted_mean":
        normalized_weights = np.asarray(normalize_weights(weights), dtype=np.float32).reshape(-1, 1, 1)
        return np.sum(stack * normalized_weights, axis=0, dtype=np.float32)
    if aggregation == "mean":
        return np.mean(stack, axis=0, dtype=np.float32)
    if aggregation == "hard_vote":
        return np.mean((stack >= float(member_threshold)).astype(np.float32), axis=0, dtype=np.float32)
    if aggregation == "max_prob":
        return np.max(stack, axis=0)
    raise AssertionError("unreachable")
