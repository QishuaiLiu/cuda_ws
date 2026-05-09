import pathlib
import sys

# project_1/tests/conftest.py -> repo root is parents[2]
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build" / "project_1"

if not BUILD_DIR.exists():
    raise RuntimeError(
        f"Expected built module under {BUILD_DIR}. "
        "Run `cmake --build build --target _project_1_py` first."
    )

sys.path.insert(0, str(BUILD_DIR))
