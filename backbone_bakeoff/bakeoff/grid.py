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
    overrides: list  # ["k=v", ...] passed to child as --set
    cfg: RunConfig
    run_id: str  # config_hash


def _kv(k, v):
    """-> "k=v" that load_config parses back to v (str raw, rest JSON ⊂ YAML)."""
    return f"{k}={v if isinstance(v, str) else json.dumps(v)}"


def _matches(cfg, when):
    if unknown := set(when) - set(vars(cfg)):
        raise ValueError(f"unknown rule keys: {sorted(unknown)}")
    return all(getattr(cfg, k) == v for k, v in when.items())


def expand(grid_path, overrides=()):
    """-> [Run] in grid order; unavailable (backbone, init) dropped. Bad keys/values raise here."""
    with open(grid_path) as f:
        spec = yaml.safe_load(f)
    base = os.path.join(os.path.dirname(os.path.abspath(grid_path)), spec.get("base", "default.yaml"))
    fixed = [_kv(k, v) for k, v in (spec.get("set") or {}).items()]
    keys, values = zip(*spec["grid"].items()) if spec.get("grid") else ((), ())
    rules = spec.get("rules") or []
    runs = []
    for combo in itertools.product(*values):
        grid_ovr = fixed + [_kv(k, v) for k, v in zip(keys, combo)]
        cfg = load_config(base, grid_ovr + list(overrides))
        ruled = [_kv(k, v) for r in rules if _matches(cfg, r["when"]) for k, v in r["set"].items()]
        ovr = grid_ovr + ruled + list(overrides)  # later wins: rules > grid, CLI > all
        cfg = load_config(base, ovr) if ruled else cfg
        if available(cfg.backbone, cfg.init):
            runs.append(Run(base, ovr, cfg, config_hash(cfg)))
    return runs


def child_env():
    """os.environ + PKG_ROOT on PYTHONPATH -> `python -m bakeoff.*` works without install."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [PKG_ROOT, os.environ.get("PYTHONPATH")]))}


def pending(runs):
    done = {}
    for r in runs:
        if r.cfg.results_csv not in done:
            done[r.cfg.results_csv] = done_hashes(r.cfg.results_csv)
    return [r for r in runs if r.run_id not in done[r.cfg.results_csv]]


def _last_line(log):
    with open(log) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines[-1] if lines else ""


def _log_failure(r, returncode, log, error):
    append_csv(os.path.join(r.cfg.out_dir, "failures.csv"), pd.DataFrame([dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"), config_hash=r.run_id, returncode=returncode, error=error,
        overrides=json.dumps(r.overrides), log=log)]))


def launch(grid_path, overrides=(), cmd=None, dry_run=False, timeout_s=None):
    """Run pending grid -> dict(ok, failed, skipped). cmd: child argv prefix (default bakeoff.train).
    timeout_s: kill hung run -> failure."""
    runs = expand(grid_path, overrides)
    todo = pending(runs)
    out = dict(ok=0, failed=0, skipped=len(runs) - len(todo))
    print(f"{len(runs)} runs, {len(todo)} to do")
    env = child_env()
    for i, r in enumerate(todo, 1):
        c = r.cfg
        tag = f"[{i}/{len(todo)}] {c.mode} {c.backbone} {c.init} frac={c.frac} seed={c.seed} ({r.run_id})"
        if dry_run:
            print(tag)
            continue
        log = os.path.join(c.out_dir, "logs", f"{r.run_id}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        argv = [*(cmd or TRAIN_CMD), "--config", r.base, "--set", "progress=false", *r.overrides]
        with open(log, "w") as f:
            try:
                rc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout_s).returncode
                error = "" if rc == 0 else _last_line(log)
            except subprocess.TimeoutExpired:
                rc, error = -1, f"timeout after {timeout_s}s"
        out["ok" if rc == 0 else "failed"] += 1
        if rc:
            _log_failure(r, rc, log, error)
        print(tag, "ok" if rc == 0 else f"FAILED rc={rc} {error} -> {log}", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grid")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="overrides for every run")
    p.add_argument("--dry-run", action="store_true", help="list pending runs only")
    p.add_argument("--timeout-h", type=float, help="kill a run after this many hours (-> failures.csv)")
    a = p.parse_args(argv)
    out = launch(a.grid, a.set, dry_run=a.dry_run, timeout_s=a.timeout_h and a.timeout_h * 3600)
    print(out)
    return out


if __name__ == "__main__":
    main()
