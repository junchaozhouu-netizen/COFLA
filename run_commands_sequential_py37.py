#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run backslash-continued python commands from a txt file sequentially."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_commands(text: str) -> list[str]:
    commands: list[str] = []
    current: list[str] = []
    active = False

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("###"):
            continue
        if stripped.startswith("export ") or stripped.startswith("cd ") or stripped.startswith("mkdir "):
            continue

        if stripped.startswith("python ") or stripped.startswith("python3 "):
            if current:
                commands.append(" ".join(current).replace("\\ ", " ").strip())
                current = []
            active = True

        if active:
            if stripped.endswith("\\"):
                current.append(stripped[:-1].strip())
            else:
                current.append(stripped)
                commands.append(" ".join(current).replace("\\ ", " ").strip())
                current = []
                active = False

    if current:
        commands.append(" ".join(current).replace("\\ ", " ").strip())
    return commands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="txt file containing python commands")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    command_file = Path(args.file)
    if not command_file.exists():
        print(f"[ERROR] command file not found: {command_file}", file=sys.stderr)
        return 2

    commands = _parse_commands(command_file.read_text(encoding="utf-8", errors="replace"))
    if not commands:
        print(f"[ERROR] no python commands found in: {command_file}", file=sys.stderr)
        return 3

    env = os.environ.copy()
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8",
    )
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[INFO] command_file={command_file}")
    print(f"[INFO] workdir={args.workdir}")
    print(f"[INFO] num_commands={len(commands)}")

    failed = 0
    for idx, cmd in enumerate(commands, 1):
        print("\n" + "=" * 100)
        print(f"[RUN {idx}/{len(commands)}] {cmd}")
        print("=" * 100)
        if args.dry_run:
            continue
        ret = subprocess.run(cmd, shell=True, cwd=args.workdir, env=env)
        if ret.returncode != 0:
            failed += 1
            print(f"[ERROR] command {idx} exited with code {ret.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return ret.returncode
    if failed:
        print(f"[DONE WITH ERRORS] failed={failed}/{len(commands)}")
        return 1
    print("[DONE] all commands finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
