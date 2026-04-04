from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def run(target: str) -> int:
    module_name, func_name = target.split(":", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)
    return int(func())

