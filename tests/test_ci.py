"""The CI workflow exists, is valid YAML, and runs lint + tests on push and pull request."""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
README = Path(__file__).resolve().parents[1] / "README.md"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_workflow_file_exists_and_parses():
    wf = _load_workflow()
    assert wf["name"]
    assert "jobs" in wf


def test_workflow_triggers_on_push_and_pull_request():
    wf = _load_workflow()
    # PyYAML parses the bare key `on` as boolean True.
    triggers = wf.get("on", wf.get(True))
    assert "push" in triggers
    assert "pull_request" in triggers


def test_workflow_runs_ruff_and_pytest():
    wf = _load_workflow()
    steps = [step for job in wf["jobs"].values() for step in job["steps"]]
    commands = " ".join(step.get("run", "") for step in steps)
    assert "ruff check" in commands
    assert "ruff format --check" in commands
    assert "pytest" in commands


def test_readme_has_ci_badge():
    assert "workflows/ci.yml/badge.svg" in README.read_text()
