"""Experiment configuration with YAML support."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    # ---- Experiment meta ----
    experiment_name: str = "default"
    seed: int = 42

    # ---- Backbone ----
    backbone_size: str = "s"  # s | b | l
    backbone_frozen: bool = True

    # ---- Fusion ----
    fusion_type: str = "film"  # prototype_cosine | concat | film | correlation_4d

    # ---- Decoder ----
    decoder_channels: int = 256
    num_classes: int = 1  # binary segmentation

    # ---- Data ----
    img_size: int = 322  # multiple of 14 for DINOv2
    batch_size: int = 8
    num_workers: int = 0  # MPS is more stable with num_workers=0
    samples_per_epoch: int = 1000

    # ---- Training ----
    epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    scheduler: str = "cosine"  # cosine | plateau
    run_preflight: bool = True

    # ---- Loss ----
    bce_weight: float = 0.5
    dice_weight: float = 0.5

    # ---- Checkpoints ----
    checkpoint_every: int = 5

    # ---- Feature cache ----
    use_cached_features: bool = True
    cache_dtype: str = "float16"  # float16 | float32

    # ---- Camera split ----
    val_fold: int = 0

    # ---- Evaluation extensions ----
    hard_example_ids_path: str = ""

    # ---- Debug / smoke test ----
    debug: bool = False
    debug_samples: int = 8

    # ---- Derived paths (set at runtime) ----
    _project_root: str = ""

    def __post_init__(self):
        self.backbone_size = str(self.backbone_size).lower()
        if self.backbone_size not in {"s", "b", "l"}:
            raise ValueError(f"Unsupported backbone_size: {self.backbone_size}")
        if self.fusion_type not in {"prototype_cosine", "concat", "film", "correlation_4d"}:
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
        if self.img_size % 14 != 0:
            raise ValueError(f"img_size must be divisible by 14 for DINOv2, got {self.img_size}")

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def set_project_root(self, root: str | Path):
        self._project_root = str(root)

    @property
    def project_root(self) -> Path:
        return Path(self._project_root)

    @property
    def save_dir(self) -> Path:
        return self.project_root / "artifacts" / "runs" / "dinov2_research" / "experiments" / self.experiment_name

    @property
    def cache_dir(self) -> Path:
        return (
            self.project_root
            / "artifacts"
            / "runs"
            / "dinov2_research"
            / "feature_cache"
            / f"dinov2_{self.backbone_size}_{self.img_size}"
        )

    @property
    def pseudo_label_root(self) -> Path:
        return self.project_root / "artifacts" / "runs" / "dinov2_research" / "pseudo_labels"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "raw" / "dl-lab-3-product-segmentation"

    @property
    def train_images_dir(self) -> Path:
        return self.data_dir / "train" / "images"

    @property
    def train_masks_dir(self) -> Path:
        return self.data_dir / "train" / "masks"

    @property
    def unlabeled_images_dir(self) -> Path:
        return self.data_dir / "unlabeled" / "images"

    @property
    def classification_data_dir(self) -> Path:
        return self.project_root / "data" / "raw" / "dl-lab-1-image-classification"

    @property
    def segformer_artifact_root(self) -> Path:
        return self.project_root / "artifacts" / "runs" / "segformer_boundary_semisup_macos"

    @property
    def folds_summary_path(self) -> Path:
        return self.project_root / "artifacts" / "runs" / "advanced_baseline" / "folds_summary.json"

    @property
    def fold_run_dir(self) -> Path:
        return (
            self.project_root
            / "artifacts"
            / "runs"
            / "advanced_baseline"
            / "runs"
            / "fpn_resnet34"
            / f"fold_{self.val_fold}"
        )

    @property
    def val_sample_ids_path(self) -> Path:
        return self.fold_run_dir / "val_sample_ids.json"

    @property
    def results_table_path(self) -> Path:
        return self.project_root / "artifacts" / "runs" / "dinov2_research" / "results_table.csv"

    @property
    def preflight_summary_path(self) -> Path:
        return self.save_dir / "preflight_summary.json"

    @property
    def hard_example_ids_file(self) -> Path | None:
        if not self.hard_example_ids_path:
            return None
        path = Path(self.hard_example_ids_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid_keys = {field.name for field in cls.__dataclass_fields__.values() if not field.name.startswith("_")}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    # ------------------------------------------------------------------
    # Camera fold loading
    # ------------------------------------------------------------------
    def _all_camera_ips(self) -> list[str]:
        cameras = {p.stem.split("_")[0] for p in self.train_images_dir.glob("*.jpg")}
        return sorted(cameras)

    def load_camera_split(self) -> tuple[list[str], list[str]]:
        """Return (train_camera_ips, val_camera_ips) for the chosen fold.

        We prefer the full validation sample list from the advanced baseline run.
        `folds_summary.json` stores only previews for some folds, so reading it
        directly would silently drop cameras for fold 0.
        """
        all_cameras = self._all_camera_ips()

        if self.val_sample_ids_path.exists():
            payload = json.loads(self.val_sample_ids_path.read_text(encoding="utf-8"))
            sample_ids = payload["sample_ids"] if isinstance(payload, dict) else payload
            val_cameras = sorted({Path(sample_id).stem.split("_")[0] for sample_id in sample_ids})
            train_cameras = sorted(set(all_cameras) - set(val_cameras))
            if val_cameras and train_cameras:
                return train_cameras, val_cameras

        with open(self.folds_summary_path, encoding="utf-8") as f:
            folds_data = json.load(f)
        fold = folds_data["folds"][self.val_fold]
        train_preview = sorted(fold.get("train_groups_preview", []))
        val_preview = sorted(fold.get("val_groups_preview", []))
        if len(train_preview) == int(fold.get("train_group_count", len(train_preview))) and len(val_preview) == int(
            fold.get("val_group_count", len(val_preview))
        ):
            return train_preview, val_preview

        raise RuntimeError(
            "Could not recover full camera split. "
            f"Checked {self.val_sample_ids_path} and {self.folds_summary_path}."
        )

    def load_hard_example_ids(self) -> set[str]:
        path = self.hard_example_ids_file
        if path is None or not path.exists():
            return set()
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
