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


class TestContainerCommandContract:
    """The image's ENTRYPOINT/CMD contract, and CI's obedience to it.

    `docker/entrypoint.sh` ends with `exec python -m tradebot.app.cli "$@"`, so
    the container command is a bare **subcommand** — `run`, `validate-config`,
    and so on. `CMD ["run"]` in the Dockerfile is the contract stated in one
    place; anything that passes an interpreter instead lands in argparse's
    `command` positional and is rejected:

        tradebot: error: argument command: invalid choice: 'python'

    That is precisely what the Docker CI job did, and it is the kind of mistake
    that reads as correct — the string `python -m tradebot.app.cli
    validate-config` is a valid command, just not at this layer.
    """

    @pytest.fixture(scope="class")
    def dockerfile(self) -> str:
        return (ROOT / "docker" / "Dockerfile").read_text()

    @pytest.fixture(scope="class")
    def entrypoint(self) -> str:
        return (ROOT / "docker" / "entrypoint.sh").read_text()

    def test_the_entrypoint_supplies_the_interpreter(self, entrypoint):
        """The premise of the whole contract."""
        assert 'exec python -m tradebot.app.cli "$@"' in entrypoint

    def test_the_dockerfile_default_command_is_a_bare_subcommand(self, dockerfile):
        assert 'CMD ["run"]' in dockerfile

    def test_the_dockerfile_entrypoint_runs_the_script(self, dockerfile):
        assert "/usr/local/bin/entrypoint.sh" in dockerfile
        assert "ENTRYPOINT" in dockerfile

    def _container_commands(self, ci_workflow: str) -> list[str]:
        """Every `docker run ... tradebot:ci <command>` argument in CI.

        `docker compose exec` is deliberately not matched: it bypasses the
        ENTRYPOINT, so an explicit interpreter is correct there.
        """
        commands: list[str] = []
        # Join continuation lines so a wrapped `docker run` is one string.
        joined = ci_workflow.replace("\\\n", " ")
        for line in joined.splitlines():
            stripped = line.strip()
            if "docker run" not in stripped or "tradebot:ci" not in stripped:
                continue
            tokens = stripped.split()
            index = tokens.index("tradebot:ci")
            if index + 1 < len(tokens):
                commands.append(tokens[index + 1])
        return commands

    def test_ci_invokes_the_image_at_all(self, ci_workflow):
        assert self._container_commands(ci_workflow), (
            "no docker run smoke test found; the image would ship unexercised"
        )

    def test_every_ci_container_command_is_a_real_subcommand(self, ci_workflow):
        """The regression this class exists for."""
        valid = {"validate-config", "doctor", "run", "scan", "backtest", "walkforward"}
        offending = [c for c in self._container_commands(ci_workflow) if c not in valid]
        assert not offending, (
            f"CI passes {offending} as the container command, but the ENTRYPOINT "
            f"already supplies the interpreter. Pass a bare subcommand from "
            f"{sorted(valid)}."
        )

    def test_the_valid_subcommands_match_the_parser(self):
        """Keeps the list above honest if a subcommand is ever added."""
        from tradebot.app.cli import build_parser

        actions = [a for a in build_parser()._actions if getattr(a, "choices", None) is not None]
        parser_commands = set(actions[0].choices)
        assert parser_commands == {
            "validate-config",
            "doctor",
            "run",
            "scan",
            "backtest",
            "walkforward",
        }

    def test_the_smoke_test_needs_no_credentials(self, ci_workflow):
        """A smoke test that needed a secret could not run on a fork's PR."""
        joined = ci_workflow.replace("\\\n", " ")
        for line in joined.splitlines():
            if "docker run" in line and "tradebot:ci" in line:
                assert "BINANCE_API_KEY" not in line
                assert "BINANCE_API_SECRET" not in line
                assert "secrets." not in line

    def test_the_smoke_test_never_arms_live_trading(self, ci_workflow):
        """CI may prove the image REFUSES live; it must never satisfy the gate.

        The refusal cases below deliberately set LIVE, so this asserts the one
        thing that would actually be dangerous: never both the acknowledgement
        and a real-money endpoint in the same invocation.
        """
        joined = ci_workflow.replace("\\\n", " ")
        for line in joined.splitlines():
            if "docker run" not in line or "tradebot:ci" not in line:
                continue
            armed = "I_UNDERSTAND_LIVE_TRADING_RISK=YES" in line and "BINANCE_TESTNET=false" in line
            assert not armed, f"CI arms live trading: {line.strip()}"

    def test_ci_proves_the_image_refuses_live(self, ci_workflow):
        """The image's own refusal is a safety property worth a smoke test."""
        assert "TRADING_MODE=LIVE" in ci_workflow
        assert "78" in ci_workflow, "the EX_CONFIG refusal exit code is not asserted"
