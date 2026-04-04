from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def guess_project_root(start: str | Path | None = None) -> Path:
    if start is None:
        probe = Path.cwd().resolve()
    else:
        resolved = Path(start).resolve()
        probe = resolved if resolved.is_dir() else resolved.parent

    for candidate in [probe, *probe.parents]:
        if (candidate / "README.md").exists() and (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate

    raise FileNotFoundError("Could not locate project root containing README.md, src/ and configs/.")


PROJECT_ROOT = guess_project_root(Path(__file__))
DOCS_ROOT = PROJECT_ROOT / "docs"
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"
APPS_ROOT = PROJECT_ROOT / "apps"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"
DELIVERABLES_ROOT = ARTIFACTS_ROOT / "deliverables"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
LAB1_DATASET_ROOT = DATA_ROOT / "dl-lab-1-image-classification"
LAB3_DATASET_ROOT = DATA_ROOT / "dl-lab-3-product-segmentation"
CONFIGS_ROOT = PROJECT_ROOT / "configs"

