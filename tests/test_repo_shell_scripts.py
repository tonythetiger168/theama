# SPDX-License-Identifier: Apache-2.0
"""Repo-health gate for the RunPod shell entry scripts.

These scripts are the entry points executed on paid GPU pods
(`bash scripts/runpod_m2_smoke.sh /workspace` etc.). A syntax slip or a
reference to a renamed python entry point would only surface after pod
time was burned, so pin the cheap invariants here on CPU:

- every scripts/*.sh passes `bash -n`
- header contract: env-bash shebang, SPDX line, `set -euo pipefail`
- wiring: every `scripts/*.py|sh` token referenced by a shell script
  (code or usage comment) still exists in the repo
- wiring: every `python -m theama.*` module reference maps to a real
  file under src/
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

EXPECTED = {
    "download_weights.sh",
    "run_inference.sh",
    "runpod_bootstrap.sh",
    "runpod_m1_eval.sh",
    "runpod_m2_smoke.sh",
    "runpod_issue4_vbench.sh",
}

SCRIPT_REF = re.compile(r"scripts/[A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:py|sh)")
MODULE_REF = re.compile(r"python3?\s+-m\s+(theama\.[A-Za-z0-9_.]+)")


def _shell_scripts():
    return sorted(SCRIPTS_DIR.glob("*.sh"))


def test_expected_scripts_present():
    # The known five must exist; the glob must not silently shrink to zero.
    names = {p.name for p in _shell_scripts()}
    assert EXPECTED <= names


def test_bash_syntax():
    if shutil.which("bash") is None:
        pytest.skip("bash not available on this host")
    for script in _shell_scripts():
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{script.name}: {result.stderr.strip()}"


def test_header_contract():
    for script in _shell_scripts():
        lines = script.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "#!/usr/bin/env bash", script.name
        assert "SPDX-License-Identifier: Apache-2.0" in lines[1], script.name
        assert "set -euo pipefail" in lines, script.name


def test_referenced_scripts_exist():
    for script in _shell_scripts():
        text = script.read_text(encoding="utf-8")
        for ref in sorted(set(SCRIPT_REF.findall(text))):
            assert (REPO_ROOT / ref).is_file(), (
                f"{script.name} references missing {ref}"
            )


def test_python_module_refs_exist():
    for script in _shell_scripts():
        text = script.read_text(encoding="utf-8")
        for module in sorted(set(MODULE_REF.findall(text))):
            target = REPO_ROOT / "src" / (module.replace(".", "/") + ".py")
            assert target.is_file(), (
                f"{script.name} references missing module {module}"
            )
