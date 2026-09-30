"""Lightweight checks that do not require a dataset or a model service."""

from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        ROOT / "linux" / "experiment1.py",
        ROOT / "linux" / "experiment1_small.py",
        ROOT / "prompts" / "intent_classification.txt",
        ROOT / "prompts" / "action_decision.txt",
        ROOT / "configs" / "experiment.yaml",
        ROOT / "requirements-lock.txt",
        ROOT / "LICENSE",
        ROOT / "README.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert not missing, f"Missing required files: {missing}"


def test_python_sources_parse():
    sources = list((ROOT / "linux").glob("*.py"))
    assert sources
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
