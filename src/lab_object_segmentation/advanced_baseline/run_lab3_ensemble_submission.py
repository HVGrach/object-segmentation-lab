from __future__ import annotations

import argparse
import csv
import json
import random
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from lab_object_segmentation.common.paths import LAB3_DATASET_ROOT, PROJECT_ROOT, RUNS_ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_SPECS = [
    {
        "name": "unetpp_resnet34",
        "model_name": "UnetPlusPlus",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
    },
    {
        "name": "fpn_resnet34",
        "model_name": "FPN",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
    },
    {
        "name": "unet_resnet34",
        "model_name": "Unet",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
    },
]
DEFAULT_MODEL_SPEC = MODEL_SPECS[0]
N_FOLDS = 3
NUM_CLASSES = 1
ACTIVATION = None
DEFAULT_THRESHOLD = 0.50
POSTPROCESS_MIN_COMPONENT_AREA_RATIO = 0.0005
POSTPROCESS_CLOSE_KERNEL = 5


DATA_ROOT = LAB3_DATASET_ROOT
TEST_IMAGES_DIR = DATA_ROOT / "test_images"
SAVE_DIR = RUNS_ROOT / "advanced_baseline"


def get_device() -> torch.device:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def save_png_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError(f"Failed to encode mask for {path}")
    encoded.tofile(str(path))


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def collect_image_paths(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def model_root_dir(model_spec: dict) -> Path:
    return SAVE_DIR / "runs" / model_spec["name"]


def fold_run_dir(model_spec: dict, fold_idx: int) -> Path:
    return model_root_dir(model_spec) / f"fold_{fold_idx}"


def relative_output_paths(input_root: Path, output_root: Path, image_path: Path) -> tuple[Path, Path]:
    relative_path = image_path.relative_to(input_root)
    mask_path = (output_root / "masks" / relative_path).with_suffix(".png")
    prob_path = (output_root / "probs" / relative_path).with_suffix(".npy")
    return mask_path, prob_path


def autocast_context(device: torch.device, use_amp: bool):
    if not use_amp:
        return nullcontext()
    if device.type == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_model(model_spec: dict | None = None) -> nn.Module:
    model_spec = DEFAULT_MODEL_SPEC if model_spec is None else model_spec
    model_name = model_spec["model_name"]
    encoder_name = model_spec["encoder_name"]
    encoder_weights = model_spec["encoder_weights"]

    kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": 3,
        "classes": NUM_CLASSES,
        "activation": ACTIVATION,
    }

    if model_name == "UnetPlusPlus":
        return smp.UnetPlusPlus(**kwargs)
    if model_name == "Unet":
        return smp.Unet(**kwargs)
    if model_name == "FPN":
        return smp.FPN(**kwargs)
    raise ValueError(f"Unsupported model_name: {model_name}")


def load_run_model(run_dir: Path):
    checkpoint_path = run_dir / "best_checkpoint.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    current_run = checkpoint_config.get("current_run", {})
    model_spec = current_run.get("model_spec", DEFAULT_MODEL_SPEC)

    encoder_name = model_spec.get("encoder_name", DEFAULT_MODEL_SPEC["encoder_name"])
    encoder_weights = model_spec.get("encoder_weights", DEFAULT_MODEL_SPEC["encoder_weights"])
    img_size = checkpoint_config.get("train", {}).get("img_size", 384)
    best_threshold = checkpoint.get("best_threshold")
    if best_threshold is None:
        best_threshold = checkpoint_config.get("best_threshold", DEFAULT_THRESHOLD)

    inference_spec = dict(model_spec)
    inference_spec["encoder_weights"] = None
    model = build_model(inference_spec)

    state_dict_key = "ema_state_dict" if checkpoint.get("ema_state_dict") is not None else "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key])
    model.to(DEVICE)
    model.eval()

    preprocess_input = None
    if encoder_weights is not None:
        preprocess_input = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)

    return model, preprocess_input, int(img_size), float(best_threshold)


def preprocess_rgb_image(image_rgb: np.ndarray, img_size: int, preprocess_input):
    image_resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    image_resized = image_resized.astype(np.float32)

    if preprocess_input is not None:
        image_resized = preprocess_input(image_resized)
    else:
        image_resized = image_resized / 255.0

    tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(DEVICE)


@torch.no_grad()
def predict_single_model_probability_map(
    model: nn.Module,
    image_rgb: np.ndarray,
    img_size: int,
    preprocess_input,
    use_amp: bool,
    use_tta: bool = True,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]

    inputs = [image_rgb]
    if use_tta:
        inputs.append(image_rgb[:, ::-1].copy())

    probability_maps = []
    for index, image_variant in enumerate(inputs):
        tensor = preprocess_rgb_image(image_variant, img_size, preprocess_input)
        with autocast_context(DEVICE, use_amp):
            logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        if index == 1:
            probs = probs[:, ::-1]
        probability_maps.append(probs)

    mean_probs = np.mean(probability_maps, axis=0)
    if mean_probs.shape != (height, width):
        mean_probs = cv2.resize(mean_probs.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return mean_probs.astype(np.float32)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    cleaned = np.zeros_like(mask_uint8)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_idx] = 1

    return cleaned


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    min_area = max(16, int(mask.size * POSTPROCESS_MIN_COMPONENT_AREA_RATIO))
    cleaned = remove_small_components(mask, min_area=min_area)

    if POSTPROCESS_CLOSE_KERNEL > 1:
        kernel = np.ones((POSTPROCESS_CLOSE_KERNEL, POSTPROCESS_CLOSE_KERNEL), dtype=np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = (cleaned > 0).astype(np.uint8)

    return cleaned


def load_ensemble_members(ensemble_summary_path: Path):
    ensemble_summary = load_json(ensemble_summary_path)
    if not ensemble_summary:
        raise FileNotFoundError(f"Ensemble summary does not exist: {ensemble_summary_path}")

    members = []
    for model_name, model_weight in ensemble_summary["weights_by_model"].items():
        model_spec = next(spec for spec in MODEL_SPECS if spec["name"] == model_name)

        available_folds = []
        for fold_idx in range(N_FOLDS):
            run_dir = fold_run_dir(model_spec, fold_idx)
            if (run_dir / "best_checkpoint.pth").exists():
                available_folds.append((fold_idx, run_dir))

        if not available_folds:
            continue

        per_fold_weight = float(model_weight) / len(available_folds)
        for fold_idx, run_dir in available_folds:
            model, preprocess_input, img_size, _ = load_run_model(run_dir)
            members.append(
                {
                    "model_name": model_name,
                    "fold_idx": fold_idx,
                    "weight": per_fold_weight,
                    "model": model,
                    "preprocess_input": preprocess_input,
                    "img_size": img_size,
                }
            )

    return ensemble_summary, members


@torch.no_grad()
def predict_ensemble_probability_map(
    members: list[dict],
    image_rgb: np.ndarray,
    use_amp: bool,
    use_tta: bool = True,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    combined = np.zeros((height, width), dtype=np.float32)

    for member in members:
        probs = predict_single_model_probability_map(
            model=member["model"],
            image_rgb=image_rgb,
            img_size=member["img_size"],
            preprocess_input=member["preprocess_input"],
            use_amp=use_amp,
            use_tta=use_tta,
        )
        combined += float(member["weight"]) * probs

    return combined.astype(np.float32)


def serialize_mask(mask2d: np.ndarray) -> str:
    return json.dumps(mask2d.astype(np.uint8).tolist(), separators=(",", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trained ensemble on lab3 test images and build a Kaggle submission.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TEST_IMAGES_DIR,
        help="Directory with test images.",
    )
    parser.add_argument(
        "--ensemble-summary",
        type=Path,
        default=SAVE_DIR / "ensemble_summary.json",
        help="Path to ensemble_summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAVE_DIR / "ensemble_outputs" / "test_images_lab3",
        help="Where to save predicted masks and optional probability maps.",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=SAVE_DIR / "submission_lab3_test_images.csv",
        help="Where to save the submission CSV.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SAVE_DIR / "submission_lab3_test_images_summary.json",
        help="Where to save a compact run summary.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override ensemble threshold. By default uses best_threshold from ensemble_summary.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N images for a smoke test.",
    )
    parser.add_argument(
        "--save-probability-maps",
        action="store_true",
        help="Also save .npy probability maps.",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable horizontal-flip TTA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    image_paths = collect_image_paths(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {args.input_dir}")

    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    ensemble_summary, members = load_ensemble_members(args.ensemble_summary)
    if not members:
        raise RuntimeError("No ensemble members were loaded.")

    threshold = ensemble_summary["best_threshold"] if args.threshold is None else float(args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device          : {DEVICE}")
    print(f"Images          : {len(image_paths)}")
    print(f"Ensemble members: {len(members)}")
    print(f"Threshold       : {threshold:.2f}")
    print(f"TTA             : {not args.no_tta}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Submission path : {args.submission_path}")

    stats_rows: list[dict] = []
    with args.submission_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ImageId", "mask"])

        for index, image_path in enumerate(image_paths, 1):
            image_bgr = read_image(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Cannot read image: {image_path}")

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            probabilities = predict_ensemble_probability_map(
                members=members,
                image_rgb=image_rgb,
                use_amp=False,
                use_tta=not args.no_tta,
            )
            mask = postprocess_mask((probabilities >= threshold).astype(np.uint8))

            mask_path, prob_path = relative_output_paths(
                input_root=args.input_dir,
                output_root=args.output_dir,
                image_path=image_path,
            )
            save_png_mask(mask_path, mask.astype(np.uint8) * 255)
            if args.save_probability_maps:
                prob_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(prob_path, probabilities.astype(np.float16))

            writer.writerow([image_path.name, serialize_mask(mask)])
            stats_rows.append(
                {
                    "image_name": image_path.name,
                    "relative_path": str(image_path.relative_to(args.input_dir)),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "prob_path": str(prob_path) if args.save_probability_maps else None,
                    "area_ratio": float(mask.mean()),
                    "mean_probability": float(probabilities.mean()),
                }
            )

            if index % 50 == 0 or index == len(image_paths):
                print(f"Processed {index}/{len(image_paths)}")

    save_json(
        args.summary_path,
        {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "submission_path": str(args.submission_path),
            "num_images": len(stats_rows),
            "threshold": threshold,
            "tta_enabled": not args.no_tta,
            "save_probability_maps": bool(args.save_probability_maps),
            "device": str(DEVICE),
            "selected_models": ensemble_summary["selected_models"],
            "weights_by_model": ensemble_summary["weights_by_model"],
            "stats_preview": stats_rows[:5],
        },
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
