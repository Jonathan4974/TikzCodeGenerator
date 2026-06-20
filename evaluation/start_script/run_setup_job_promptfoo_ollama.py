#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_USER = "s0030"
DEFAULT_MODEL = "gemma4:e2b-it-qat"
DEFAULT_TYPE = "img-tikz"

SCRIPT_DIR = Path(__file__).resolve().parent
SBATCH_SCRIPT = SCRIPT_DIR / "ollama_promptfoo.sbatch"


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}")
    print(f"cwd={cwd}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run setup scripts and submit the Promptfoo/Ollama Slurm job."
    )

    parser.add_argument("--user", default=os.environ.get("I9_USER", DEFAULT_USER))
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--type", default=os.environ.get("TYPE", DEFAULT_TYPE))
    parser.add_argument(
        "--promptfoo-config",
        required=True,
        help="Absolute path to the Promptfoo config YAML.",
    )

    parser.add_argument("--skip-latex", action="store_true")
    parser.add_argument("--skip-envs", action="store_true")
    parser.add_argument("--no-submit", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project = Path(f"/usr/prakt/{args.user}/projects/tikzcodegenerator")
    promptfoo_config = Path(args.promptfoo_config)

    setup_latex = SCRIPT_DIR / "setup_latex.sh"
    setup_envs = SCRIPT_DIR / "setup_envs.sh"

    if not project.is_dir():
        raise SystemExit(f"Project directory does not exist: {project}")

    if not setup_latex.is_file():
        raise SystemExit(f"setup_latex.sh not found: {setup_latex}")

    if not setup_envs.is_file():
        raise SystemExit(f"setup_envs.sh not found: {setup_envs}")

    if not promptfoo_config.is_absolute():
        raise SystemExit("--promptfoo-config must be an absolute path")

    if not promptfoo_config.is_file():
        raise SystemExit(f"Promptfoo config does not exist: {promptfoo_config}")

    env = os.environ.copy()
    env["I9_USER"] = args.user
    env["MODEL"] = args.model
    env["TYPE"] = args.type
    env["PROMPTFOO_CONFIG"] = str(promptfoo_config)

    print(f"I9_USER={args.user}")
    print(f"PROJECT={project}")
    print(f"MODEL={args.model}")
    print(f"TYPE={args.type}")
    print(f"PROMPTFOO_CONFIG={promptfoo_config}")

    steps: list[tuple[bool, list[str]]] = [
        (not args.skip_latex, ["bash", str(setup_latex)]),
        (not args.skip_envs, ["bash", str(setup_envs)]),
        (not args.no_submit, ["sbatch", str(SBATCH_SCRIPT)]),
    ]

    for enabled, cmd in steps:
        if enabled:
            run(cmd, cwd=project, env=env)

    print("\nDone.")


if __name__ == "__main__":
    main()