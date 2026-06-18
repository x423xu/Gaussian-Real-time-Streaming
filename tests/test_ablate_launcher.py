from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def run_dry_run(free_memory: str = "9:48665,0:29335,1:36511,2:3880,4:31815") -> str:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "GPU_FREE_MEMORY": free_memory,
            "MIN_FREE_VRAM_MB": "10000",
        }
    )
    result = subprocess.run(
        ["bash", "ablate_train.sh"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout


def test_ablate_launcher_assigns_parallel_gpus_with_10gb_capacity() -> None:
    output = run_dry_run()
    launch_lines = [line for line in output.splitlines() if line.startswith("[DRY-RUN]")]

    assert len(launch_lines) == 11
    assert launch_lines[0].startswith("[DRY-RUN] base gpu=9")

    assigned_gpus = [re.search(r"gpu=(\d+)", line).group(1) for line in launch_lines]
    counts = Counter(assigned_gpus)
    assert len(counts) > 1
    assert "2" not in counts
    assert counts["9"] <= 4
    assert counts["0"] <= 2
    assert counts["1"] <= 3
    assert counts["4"] <= 3


def test_ablate_launcher_uses_unique_output_dirs_and_wandb_names() -> None:
    output = run_dry_run()
    output_dirs = re.findall(r"output_dir=([^\s]+)", output)
    wandb_names = re.findall(r"wandb.name=([^\s]+)", output)

    assert len(output_dirs) == 11
    assert len(wandb_names) == 11
    assert len(set(output_dirs)) == 11
    assert len(set(wandb_names)) == 11
    assert "outputs/rtgs_ablate_base" in output_dirs
    assert "rtgs_ablate_all_refinements_train_depth_head_only" in wandb_names


def test_ablate_launcher_uses_rtgs_conda_python_by_default() -> None:
    output = run_dry_run()

    assert "/data0/xxy/miniconda3/bin/conda" in output
    assert "--no-capture-output" in output
    assert "-n rtgs python -m rtgs.main" in output
