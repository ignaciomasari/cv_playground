"""Grid launcher: one subprocess per run, skip finished (config_hash in results), failures logged + skipped.

`python -m bakeoff.grid configs/grid.yaml [--set epochs=2] [--dry-run]`
grid.yaml: base (config path, rel. to grid file), set (fixed overrides), grid (key -> list; cartesian product),
rules (optional [{when: {backbone: .., init: ..}, set: {..}}] -> overrides for matching runs, e.g. tuned recipes).
"""
import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

import pandas as pd
import yaml

from . import setup_logging
from .config import RunConfig, config_hash, load_config
from .models import available
from .results import KEY, append_csv, done_hashes

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for PYTHONPATH of children
TRAIN_CMD = [sys.executable, "-m", "bakeoff.train"]
log = logging.getLogger("bakeoff.grid")  # not __name__: "__main__" under `python -m`


@dataclass
class Run:
    base: str
    overrides: list[str]  # ["key=value", ...] passed to child as --set
    cfg: RunConfig
    run_id: str  # config_hash


def plain(value: Any) -> Any:
    """Enum -> its value (YAML-safe)."""
    return value.value if isinstance(value, Enum) else value


def _override(key: str, value: Any) -> str:
    """-> "key=value" that load_config parses back to value (str raw, rest JSON ⊂ YAML)."""
    return f"{key}={value if isinstance(value, str) else json.dumps(value)}"


@dataclass
class Rule:
    """Overrides `set` for runs whose cfg matches all of `when`."""
    when: dict[str, Any]
    set: dict[str, Any]

    def matches(self, cfg: RunConfig) -> bool:
        if unknown := set(self.when) - set(vars(cfg)):
            raise ValueError(f"unknown rule keys: {sorted(unknown)}")
        return all(getattr(cfg, key) == value for key, value in self.when.items())


@dataclass
class GridSpec:
    """Grid YAML: base (config path, abs after load), set (fixed overrides), grid (key -> values), rules."""
    base: str
    set: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, list] = field(default_factory=dict)
    rules: list[Rule] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "GridSpec":
        with open(path) as file:
            spec = yaml.safe_load(file)
        if unknown := set(spec) - {"base", "set", "grid", "rules"}:
            raise ValueError(f"unknown grid keys in {path}: {sorted(unknown)}")
        base = os.path.join(os.path.dirname(os.path.abspath(path)), spec.get("base", "default.yaml"))
        return cls(base, spec.get("set") or {}, spec.get("grid") or {},
                   [Rule(rule["when"], rule["set"]) for rule in spec.get("rules") or []])

    def dump(self, path: str, header: str = "") -> None:
        spec = dict(base=self.base, set=self.set, grid=self.grid,
                    rules=[dict(when={k: plain(v) for k, v in rule.when.items()},
                                set={k: plain(v) for k, v in rule.set.items()}) for rule in self.rules])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as file:
            file.write(header)
            yaml.safe_dump(spec, file, sort_keys=False)

    @property
    def swept_keys(self) -> list[str]:
        """Grid keys that are hparams (not run keys like backbone / seed)."""
        return [key for key in self.grid if key not in KEY]

    def with_rules(self, rules: Sequence[Rule]) -> "GridSpec":
        """Copy + rules appended (later wins)."""
        return replace(self, rules=[*self.rules, *rules])

    def expand(self, overrides: Sequence[str] = ()) -> list[Run]:
        """-> [Run] in grid order; unavailable (backbone, init) dropped. Bad keys/values raise here."""
        fixed = [_override(key, value) for key, value in self.set.items()]
        keys, value_lists = zip(*self.grid.items()) if self.grid else ((), ())
        runs = []
        for combo in itertools.product(*value_lists):
            grid_overrides = fixed + [_override(key, value) for key, value in zip(keys, combo)]
            cfg = load_config(self.base, grid_overrides + list(overrides))
            rule_overrides = [_override(key, value) for rule in self.rules if rule.matches(cfg)
                              for key, value in rule.set.items()]
            all_overrides = grid_overrides + rule_overrides + list(overrides)  # later wins: rules > grid, CLI > all
            cfg = load_config(self.base, all_overrides) if rule_overrides else cfg
            if available(cfg.backbone, cfg.init):
                runs.append(Run(self.base, all_overrides, cfg, config_hash(cfg)))
        return runs


def expand(grid_path: str, overrides: Sequence[str] = ()) -> list[Run]:
    return GridSpec.load(grid_path).expand(overrides)


def child_env() -> dict[str, str]:
    """os.environ + PKG_ROOT on PYTHONPATH -> `python -m bakeoff.*` works without install."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [PKG_ROOT, os.environ.get("PYTHONPATH")]))}


def pending(runs: Sequence[Run]) -> list[Run]:
    done = {}
    for run in runs:
        if run.cfg.results_csv not in done:
            done[run.cfg.results_csv] = done_hashes(run.cfg.results_csv)
    return [run for run in runs if run.run_id not in done[run.cfg.results_csv]]


def _last_line(log_path: str) -> str:
    with open(log_path) as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines[-1] if lines else ""


def _log_failure(run: Run, returncode: int, log_path: str, error: str) -> None:
    append_csv(os.path.join(run.cfg.out_dir, "failures.csv"), pd.DataFrame([dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"), config_hash=run.run_id, returncode=returncode, error=error,
        overrides=json.dumps(run.overrides), log=log_path)]))


def launch(grid_path: str, overrides: Sequence[str] = (), cmd: Sequence[str] | None = None, dry_run: bool = False,
           timeout_s: float | None = None) -> dict[str, int]:
    """Run pending grid -> dict(ok, failed, skipped). cmd: child argv prefix (default bakeoff.train).
    timeout_s: kill hung run -> failure."""
    runs = expand(grid_path, overrides)
    todo = pending(runs)
    counts = dict(ok=0, failed=0, skipped=len(runs) - len(todo))
    log.info("%d runs, %d to do", len(runs), len(todo))
    env = child_env()
    for number, run in enumerate(todo, 1):
        cfg = run.cfg
        label = (f"[{number}/{len(todo)}] {cfg.mode} {cfg.backbone} {cfg.init} "
                 f"frac={cfg.frac} seed={cfg.seed} ({run.run_id})")
        if dry_run:
            log.info(label)
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
        if returncode == 0:
            log.info("%s ok", label)
        else:
            log.warning("%s FAILED rc=%s %s -> %s", label, returncode, error, log_path)
    return counts


def main(argv: Sequence[str] | None = None) -> dict[str, int]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("grid")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="overrides for every run")
    parser.add_argument("--dry-run", action="store_true", help="list pending runs only")
    parser.add_argument("--timeout-h", type=float, help="kill a run after this many hours (-> failures.csv)")
    args = parser.parse_args(argv)
    setup_logging()
    counts = launch(args.grid, args.set, dry_run=args.dry_run,
                    timeout_s=args.timeout_h and args.timeout_h * 3600)
    print(counts)
    return counts


if __name__ == "__main__":
    main()
