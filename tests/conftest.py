"""Make the workspace's source and notebook builders importable during tests."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))
