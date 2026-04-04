#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, RUNS_ROOT
from lab_object_segmentation.supervised_v4.train import (
    TrainConfig,
    SegFormerV4,
    build_val_transform,
    empty_device_cache,
    get_device,
    postprocess_binary_mask,
    predict_with_tta,
    upsample_logits,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_INPUT_DIR = LAB3_DATASET_ROOT / "test_images"
DEFAULT_RUN_ROOT = RUNS_ROOT / "supervised_v4"

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


@dataclass
class EnsembleMember:
    run_name: str
    weight: float
    run_dir: Path
    config: TrainConfig
    model: SegFormerV4
    transform: object


def collect_image_paths(input_dir: Path) -> list[Path]:
    return sorted([path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS])


def read_image_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def serialize_mask(mask2d: np.ndarray) -> str:
    return json.dumps(mask2d.astype(np.uint8).tolist(), separators=(",", ":"))


def relative_output_paths(input_root: Path, output_root: Path, image_path: Path) -> tuple[Path, Path]:
    relative_path = image_path.relative_to(input_root)
    mask_path = (output_root / "masks" / relative_path).with_suffix(".png")
    prob_path = (output_root / "probs" / relative_path).with_suffix(".npy")
    return mask_path, prob_path


def threshold_tag(threshold: float) -> str:
    return f"thr{int(round(threshold * 100)):02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a supervised-v4 SegFormer ensemble on test images and build a submission.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--preset", choices=sorted(PRESET_MEMBERS.keys()), default="blend4")
    parser.add_argument("--run-names", nargs="+", default=None, help="Optional explicit run names instead of preset members.")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Optional weights for --run-names.")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-tta", action="store_true", help="Disable member-side TTA.")
    parser.add_argument("--disable-postprocess", action="store_true", help="Disable final mask post-processing.")
    parser.add_argument("--postprocess-min-component-area", type=int, default=128)
    parser.add_argument("--disable-postprocess-fill-holes", action="store_true")
    parser.add_argument("--save-probability-maps", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def resolve_member_specs(args: argparse.Namespace) -> list[tuple[str, float]]:
    if args.run_names is None:
        return list(PRESET_MEMBERS[args.preset])

    run_names = [str(name) for name in args.run_names]
    if args.weights is None:
        weights = [1.0 / len(run_names)] * len(run_names)
    else:
        if len(args.weights) != len(run_names):
            raise ValueError("weights must have the same length as run-names")
        weights = [float(weight) for weight in args.weights]
        total = sum(weights)
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        weights = [weight / total for weight in weights]
    return list(zip(run_names, weights))


def load_member(run_root: Path, run_name: str, weight: float, device: torch.device) -> EnsembleMember:
    run_dir = run_root / run_name
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "best_model.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json for run: {run_name}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pth for run: {run_name}")

    config = TrainConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    model = SegFormerV4(config.backbone).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["ema_model_state_dict"] if checkpoint.get("ema_model_state_dict") is not None else checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = build_val_transform(config.image_size, config.image_mean, config.image_std)
    return EnsembleMember(
        run_name=run_name,
        weight=float(weight),
        run_dir=run_dir,
        config=config,
        model=model,
        transform=transform,
    )


@torch.no_grad()
def predict_member_probability(member: EnsembleMember, image_rgb: np.ndarray, device: torch.device, use_tta: bool) -> np.ndarray:
    transformed = member.transform(image=image_rgb)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    if use_tta:
        probs = predict_with_tta(member.model, image_tensor, image_tensor.shape[-2:], member.config)
    else:
        seg_logits_low, _ = member.model(image_tensor)
        seg_logits = upsample_logits(seg_logits_low, image_tensor.shape[-2:])
        probs = torch.sigmoid(seg_logits)

    probability_map = probs.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    orig_h, orig_w = image_rgb.shape[:2]
    if probability_map.shape != (orig_h, orig_w):
        probability_map = cv2.resize(probability_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return probability_map.astype(np.float32)


def main() -> int:
    args = parse_args()
    device = get_device()
    member_specs = resolve_member_specs(args)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    image_paths = collect_image_paths(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    members = [load_member(args.run_root, run_name, weight, device) for run_name, weight in member_specs]
    weight_sum = sum(member.weight for member in members)
    if weight_sum <= 0:
        raise ValueError("Ensemble weights must sum to a positive value")
    for member in members:
        member.weight /= weight_sum

    if args.output_dir is None:
        args.output_dir = args.run_root / "ensemble_outputs" / f"{args.preset}_{threshold_tag(args.threshold)}"
    if args.submission_path is None:
        args.submission_path = args.run_root / f"submission_supervised_v4_{args.preset}_{threshold_tag(args.threshold)}.csv"
    if args.summary_path is None:
        args.summary_path = args.run_root / f"submission_supervised_v4_{args.preset}_{threshold_tag(args.threshold)}_summary.json"

    postprocess_config = members[0].config
    postprocess_config.postprocess_enabled = not args.disable_postprocess
    postprocess_config.postprocess_min_component_area = int(args.postprocess_min_component_area)
    postprocess_config.postprocess_fill_holes = not args.disable_postprocess_fill_holes

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Images          : {len(image_paths)}")
    print(f"Preset          : {args.preset}")
    print(f"Threshold       : {args.threshold:.2f}")
    print(f"TTA             : {not args.no_tta}")
    print(f"Post-process    : {postprocess_config.postprocess_enabled}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Submission path : {args.submission_path}")
    print("Members:")
    for member in members:
        print(f"  - {member.run_name} | weight={member.weight:.4f} | image_size={member.config.image_size}")

    stats_rows: list[dict] = []
    with args.submission_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ImageId", "mask"])

        for index, image_path in enumerate(tqdm(image_paths, desc="Supervised-v4 ensemble inference"), 1):
            image_rgb = read_image_rgb(image_path)
            ensemble_prob = np.zeros(image_rgb.shape[:2], dtype=np.float32)

            member_means = {}
            for member in members:
                probability_map = predict_member_probability(member, image_rgb, device, use_tta=not args.no_tta)
                ensemble_prob += member.weight * probability_map
                member_means[member.run_name] = float(probability_map.mean())

            mask = (ensemble_prob >= float(args.threshold)).astype(np.uint8)
            if postprocess_config.postprocess_enabled:
                mask = postprocess_binary_mask(mask, postprocess_config)

            mask_path, prob_path = relative_output_paths(
                input_root=args.input_dir,
                output_root=args.output_dir,
                image_path=image_path,
            )
            save_mask_png(mask_path, mask)
            if args.save_probability_maps:
                prob_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(prob_path, ensemble_prob.astype(np.float16))

            writer.writerow([image_path.name, serialize_mask(mask)])
            stats_rows.append(
                {
                    "image_name": image_path.name,
                    "relative_path": str(image_path.relative_to(args.input_dir)),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "prob_path": str(prob_path) if args.save_probability_maps else None,
                    "area_ratio": float(mask.mean()),
                    "mean_probability": float(ensemble_prob.mean()),
                    "member_mean_probabilities": member_means,
                }
            )

            if index % 25 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)}")
                empty_device_cache(device)

    save_json(
        args.summary_path,
        {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "submission_path": str(args.submission_path),
            "num_images": len(stats_rows),
            "threshold": float(args.threshold),
            "tta_enabled": not args.no_tta,
            "postprocess_enabled": postprocess_config.postprocess_enabled,
            "postprocess_min_component_area": int(postprocess_config.postprocess_min_component_area),
            "postprocess_fill_holes": bool(postprocess_config.postprocess_fill_holes),
            "device": str(device),
            "members": [
                {
                    "run_name": member.run_name,
                    "weight": member.weight,
                    "run_dir": str(member.run_dir),
                    "image_size": member.config.image_size,
                    "mask_loss": member.config.mask_loss,
                    "tta_scales": member.config.tta_scales,
                }
                for member in members
            ],
            "stats_preview": stats_rows[:5],
        },
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
