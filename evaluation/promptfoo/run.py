import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from config import PATHS, PROMPTFOO, ROOT, apply_runtime_environment


def promptfoo_command() -> list[str]:
    local = ROOT / "node_modules" / ".bin" / ("promptfoo.cmd" if sys.platform == "win32" else "promptfoo")
    if local.exists():
        return [str(local)]

    executable = shutil.which("promptfoo")
    if executable:
        return [executable]

    npx = shutil.which("npx")
    if npx:
        return [npx, "--no-install", "promptfoo"]

    raise SystemExit("Promptfoo fehlt. Fuehre zuerst `npm install` aus.")


def validate_inputs() -> None:
    missing = [path for path in (PATHS.manifest, PATHS.crystalbleu_corpus) if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Konfiguriere die Datenpfade in config.py. Fehlend:\n{joined}")


def run(command: str, extra: list[str]) -> int:
    apply_runtime_environment()
    promptfoo = promptfoo_command()

    if command == "eval":
        validate_inputs()
        cmd = promptfoo + [
            "eval",
            "-c",
            str(PROMPTFOO.config_file),
            "-j",
            str(PROMPTFOO.max_concurrency),
            *extra,
        ]
    else:
        cmd = promptfoo + ["view", "--port", str(PROMPTFOO.view_port), "--no", *extra]

    return subprocess.run(cmd, cwd=Path(__file__).resolve().parent).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("eval", "view"))
    args, extra = parser.parse_known_args()
    return run(args.command, extra)


if __name__ == "__main__":
    raise SystemExit(main())
