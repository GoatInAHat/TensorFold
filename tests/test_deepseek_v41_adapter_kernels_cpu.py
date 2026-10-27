"""The adapter delta kernels' logic, run by Triton's CPU interpreter in a process of its own (TRITON_INTERPRET is read
when a kernel is defined, so it cannot be set inside a test session that also holds compiled kernels). The same checks
run compiled on a GPU in tests/cuda/test_deepseek_v41_adapter_kernels.py."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"


def test_the_kernels_match_the_definition_and_keep_every_row_independent_under_the_cpu_interpreter():
    code = ("import json, sys; import adapter_kernel_checks as c; "
            "print(json.dumps(c.run_all('cpu')))")
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": os.pathsep.join([str(SRC), str(TESTS)])}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900,
                          cwd=str(TESTS), check=False)
    assert done.returncode == 0, done.stderr[-3000:] + done.stdout[-1000:]
    errors = json.loads(done.stdout.strip().splitlines()[-1])
    assert set(errors) >= {"definition_bfloat16", "definition_float16", "definition_float32", "definition_wide_rank"}
    assert max(errors.values()) < 1e-3
