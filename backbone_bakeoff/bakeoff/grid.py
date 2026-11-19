"""Grid launcher: one subprocess per run, skip finished (config_hash in results), failures logged + skipped.

`python -m bakeoff.grid configs/grid.yaml [--set epochs=2] [--dry-run]`
grid.yaml: base (config path, rel. to grid file), set (fixed overrides), grid (key -> list; cartesian product),
rules (optional [{when: {backbone: .., init: ..}, set: {..}}] -> overrides for matching runs, e.g. tuned recipes).
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import pandas as pd
import yaml

from .config import RunConfig, config_hash, load_config
from .models import available
from .results import append_csv, done_hashes

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for PYTHONPATH of children
TRAIN_CMD = [sys.executable, "-m", "bakeoff.train"]


@dataclass
class Run:
    base: str
    overrides: list  # ["key=value", ...] passed to child as --set
    cfg: RunConfig
    run_id: str  # config_hash


def _override(key, value):
    """-> "key=value" that load_config parses back to value (str raw, rest JSON ⊂ YAML)."""
    return f"{key}={value if isinstance(value, str) else json.dumps(value)}"


def _matches(cfg, when):
    if unknown := set(when) - set(vars(cfg)):
        raise ValueError(f"unknown rule keys: {sorted(unknown)}")
    return all(getattr(cfg, key) == value for key, value in when.items())


def expand(grid_path, overrides=()):
    """-> [Run] in grid order; unavailable (backbone, init) dropped. Bad keys/values raise here."""
    with open(grid_path) as file:
        spec = yaml.safe_load(file)
    base = os.path.join(os.path.dirname(os.path.abspath(grid_path)), spec.get("base", "default.yaml"))
    fixed = [_override(key, value) for key, value in (spec.get("set") or {}).items()]
    keys, value_lists = zip(*spec["grid"].items()) if spec.get("grid") else ((), ())
    rules = spec.get("rules") or []
    runs = []
    for combo in itertools.product(*value_lists):
        grid_overrides = fixed + [_override(key, value) for key, value in zip(keys, combo)]
        cfg = load_config(base, grid_overrides + list(overrides))
        rule_overrides = [_override(key, value) for rule in rules if _matches(cfg, rule["when"])
                          for key, value in rule["set"].items()]
        all_overrides = grid_overrides + rule_overrides + list(overrides)  # later wins: rules > grid, CLI > all
        cfg = load_config(base, all_overrides) if rule_overrides else cfg
        if available(cfg.backbone, cfg.init):
            runs.append(Run(base, all_overrides, cfg, config_hash(cfg)))
    return runs


def child_env():
    """os.environ + PKG_ROOT on PYTHONPATH -> `python -m bakeoff.*` works without install."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [PKG_ROOT, os.environ.get("PYTHONPATH")]))}


def pending(runs):
    done = {}
    for run in runs:
        if run.cfg.results_csv not in done:
            done[run.cfg.results_csv] = done_hashes(run.cfg.results_csv)
    return [run for run in runs if run.run_id not in done[run.cfg.results_csv]]


def _last_line(log_path):
    with open(log_path) as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines[-1] if lines else ""


def _log_failure(run, returncode, log_path, error):
    append_csv(os.path.join(run.cfg.out_dir, "failures.csv"), pd.DataFrame([dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"), config_hash=run.run_id, returncode=returncode, error=error,
        overrides=json.dumps(run.overrides), log=log_path)]))


def launch(grid_path, overrides=(), cmd=None, dry_run=False, timeout_s=None):
    """Run pending grid -> dict(ok, failed, skipped). cmd: child argv prefix (default bakeoff.train).
    timeout_s: kill hung run -> failure."""
    runs = expand(grid_path, overrides)
    todo = pending(runs)
    counts = dict(ok=0, failed=0, skipped=len(runs) - len(todo))
    print(f"{len(runs)} runs, {len(todo)} to do")
    env = child_env()
    for number, run in enumerate(todo, 1):
        cfg = run.cfg
        label = (f"[{number}/{len(todo)}] {cfg.mode} {cfg.backbone} {cfg.init} "
                 f"frac={cfg.frac} seed={cfg.seed} ({run.run_id})")
        if dry_run:
            print(label)
            continue
        log_path = os.path.join(cfg.out_dir, "logs", f"{run.run_id}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        argv = [*(cmd or TRAIN_CMD), "--config", run.base, "--set", "progress=false", *run.overrides]
        with open(log_path, "w") as log_file:
            try:
                returncode = subprocess.run(argv, stdout=log_file, stderr=subprocess.STDOUT, env=env,
                                            timeout=timeout_s).returncode
                error = "" if returncode == 0 else _last_line(log_path)
            except subprocess.TimeoutExpired:
                returncode, error = -1, f"timeout after {timeout_s}s"
        counts["ok" if returncode == 0 else "failed"] += 1
        if returncode:
            _log_failure(run, returncode, log_path, error)
        print(label, "ok" if returncode == 0 else f"FAILED rc={returncode} {error} -> {log_path}", flush=True)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="overrides for every run")
    parser.add_argument("--dry-run", action="store_true", help="list pending runs only")
    parser.add_argument("--timeout-h", type=float, help="kill a run after this many hours (-> failures.csv)")
    args = parser.parse_args(argv)
    counts = launch(args.grid, args.set, dry_run=args.dry_run,
                    timeout_s=args.timeout_h and args.timeout_h * 3600)
    print(counts)
    return counts


if __name__ == "__main__":
    main()
