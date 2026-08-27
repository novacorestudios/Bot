"""Build and test environment.

These lock in fixes for AUDIT_REPORT.md C-4 and C-5. Both were the kind of
problem that is invisible until someone new clones the repository, or until a
real failure is silently swallowed by CI.
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


@pytest.fixture(scope="module")
def ci_workflow() -> str:
    return (ROOT / ".github" / "workflows" / "ci.yml").read_text()


class TestPytestRunsFromRoot:
    """C-4: `pytest` must work from a clean checkout with no PYTHONPATH."""

    def test_pythonpath_is_configured_in_pyproject(self, pyproject):
        assert pyproject["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]

    def test_the_package_imports_without_pythonpath(self):
        """The decisive check: a subprocess with PYTHONPATH stripped."""
        import os

        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", "import tradebot; print(tradebot.__version__)"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"the package does not import without PYTHONPATH: {result.stderr}"
        )

    def test_a_dev_extra_exists_for_one_command_setup(self, pyproject):
        assert "dev" in pyproject["project"]["optional-dependencies"]


class TestCIGatesAreReal:
    """C-5: a failing quality gate must fail the build."""

    def test_no_step_suppresses_its_exit_code(self, ci_workflow):
        offending = [
            line.strip()
            for line in ci_workflow.splitlines()
            if "|| true" in line and not line.strip().startswith("#")
        ]
        assert not offending, (
            f"CI steps hide their failures: {offending}. A green build must "
            f"mean the checks actually passed."
        )

    @pytest.mark.parametrize("gate", ["mypy", "bandit", "pip-audit", "check_secrets.py", "pytest"])
    def test_every_quality_gate_is_present(self, ci_workflow, gate):
        assert gate in ci_workflow

    def test_the_build_job_depends_on_lint_and_test(self, ci_workflow):
        assert "needs: [lint, test]" in ci_workflow


class TestTypeChecking:
    """L-1: the typing debt was unmeasured because mypy never ran for real."""

    def test_the_pydantic_plugin_is_configured(self, pyproject):
        """Without it, mypy reports ~139 phantom errors that hide the real ones."""
        assert "pydantic.mypy" in pyproject["tool"]["mypy"]["plugins"]

    @pytest.mark.slow
    def test_mypy_passes_cleanly(self):
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "mypy", "src/tradebot"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"mypy is a CI gate and is failing:\n{result.stdout[-3000:]}"


class TestMakefile:
    """L-4: the documented commands should be one word."""

    def test_the_makefile_exposes_the_documented_workflow(self):
        makefile = (ROOT / "Makefile").read_text()
        for target in ("install", "test", "lint", "type", "security", "check", "run", "validate"):
            assert f"{target}:" in makefile, f"missing make target: {target}"

    def test_check_runs_every_gate(self):
        makefile = (ROOT / "Makefile").read_text()
        check_line = next(line for line in makefile.splitlines() if line.startswith("check:"))
        for gate in ("lint", "type", "test", "security"):
            assert gate in check_line
