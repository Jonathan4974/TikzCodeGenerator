#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import time
from pathlib import Path


DEFAULT_USER = "s0030"
DEFAULT_MODEL = "gemma4:e2b-it-qat"
DEFAULT_TYPE = "img-tikz"

SCRIPT_DIR = Path(__file__).resolve().parent
SBATCH_SCRIPT = SCRIPT_DIR / "ollama_promptfoo.sbatch"


def run(cmd: list[str], cwd: Path, env: dict[str, str], capture: bool = False):
    print(f"\n$ {' '.join(map(str, cmd))}")
    print(f"cwd={cwd}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def make_split_config(base_config: Path, manifest: Path, out_dir: Path, split: int) -> Path:
    if not manifest.is_file():
        raise SystemExit(f"Manifest split does not exist: {manifest}")

    text = base_config.read_text(encoding="utf-8")
    text, count = re.subn(
        r"^tests:\s*file://.*$",
        f"tests: file://{manifest}",
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if count != 1:
        raise SystemExit(f"Could not replace tests line in {base_config}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{base_config.stem}_split_{split}{base_config.suffix}"
    out.write_text(text, encoding="utf-8")
    return out


def submit_job(project: Path, env: dict[str, str]) -> str:
    result = run(
        ["sbatch", "--parsable", str(SBATCH_SCRIPT)],
        cwd=project,
        env=env,
        capture=True,
    )
    job_id = result.stdout.strip().split(";")[0]
    print(f"Submitted job: {job_id}")
    return job_id


def job_state(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
        text=True,
        capture_output=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        state = line.strip().split("|")[0].split()[0]
        if state:
            return state

    return "UNKNOWN"


def wait_for_job(job_id: str, interval: int) -> None:
    done_states = {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
    }

    while True:
        state = job_state(job_id)
        print(f"Job {job_id}: {state}")

        if state in done_states:
            if state != "COMPLETED":
                raise SystemExit(f"Job {job_id} ended with state {state}")
            return

        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run setup scripts and submit Promptfoo/Ollama split jobs one after another."
    )

    parser.add_argument("--user", default=os.environ.get("I9_USER", DEFAULT_USER))
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--type", default=os.environ.get("TYPE", DEFAULT_TYPE))
    parser.add_argument("--promptfoo-config", required=True)
    parser.add_argument("--start-split", type=int, default=1)
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--wait-seconds", type=int, default=600)

    parser.add_argument("--skip-latex", action="store_true")
    parser.add_argument("--skip-envs", action="store_true")
    parser.add_argument("--no-submit", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project = Path(f"/usr/prakt/{args.user}/projects/tikzcodegenerator")
    data_dir = Path(f"/usr/prakt/{args.user}/projects/data/benchmark_data")
    promptfoo_config = Path(args.promptfoo_config)

    setup_latex = SCRIPT_DIR / "setup_latex.sh"
    setup_envs = SCRIPT_DIR / "setup_envs.sh"
    generated_config_dir = project / "evaluation" / "promptfoo" / "generated_configs"

    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")
    if not promptfoo_config.is_absolute() or not promptfoo_config.is_file():
        raise SystemExit(f"Promptfoo config must be an existing absolute path: {promptfoo_config}")
    if not SBATCH_SCRIPT.is_file():
        raise SystemExit(f"Slurm script not found: {SBATCH_SCRIPT}")

    env = os.environ.copy()
    env["I9_USER"] = args.user
    env["MODEL"] = args.model
    env["TYPE"] = args.type

    if not args.skip_latex:
        run(["bash", str(setup_latex)], cwd=project, env=env)

    if not args.skip_envs:
        run(["bash", str(setup_envs)], cwd=project, env=env)

    for split in range(args.start_split, args.num_splits + 1):
        manifest = data_dir / "manifest_splits" / f"image_manifest_{split}.csv"
        split_config = make_split_config(
            promptfoo_config,
            manifest,
            generated_config_dir,
            split,
        )

        env["PROMPTFOO_CONFIG"] = str(split_config)
        env["MANIFEST_SPLIT"] = str(split)

        print(f"\n=== Split {split}/{args.num_splits} ===")
        print(f"PROMPTFOO_CONFIG={split_config}")

        if args.no_submit:
            continue

        job_id = submit_job(project, env)
        wait_for_job(job_id, args.wait_seconds)

    print("\nDone.")


if __name__ == "__main__":
    main()