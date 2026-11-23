"""Results analysis: varied hparams (from row config JSON), seed-mean summary, paired deltas, Pareto."""
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from .config import MODE_KEYS
from .results import KEY

GROUP = [k for k in KEY if k != "seed"] + ["in_chans"]  # summary rows
log = logging.getLogger(__name__)
BENCH_COLS = ["params_M", "infer_ips", "infer_mem_MB", "train_ips", "train_mem_MB"]  # bench.csv


def hparams(df: pd.DataFrame) -> pd.DataFrame:
    """Recipe values per row from `config` JSON (no config, e.g. notebook rows -> NaN). Run keys dropped."""
    configs = [json.loads(config) if isinstance(config, str) else {} for config in df.config]
    recipe_values = pd.DataFrame(configs, index=df.index).drop(columns=KEY, errors="ignore")
    return recipe_values.map(lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value)  # hashable


def varied(df: pd.DataFrame) -> list[str]:
    """Recipe keys with >1 value within some (mode, frac) -> resolved epochs (set by frac) not counted."""
    recipe_values = hparams(df)
    if recipe_values.empty:
        return []
    n_unique = recipe_values.groupby([df["mode"], df.frac]).nunique().max()
    return [k for k in recipe_values.columns if n_unique[k] > 1]


def with_varied(df: pd.DataFrame, split_by: Sequence[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """-> (df + split_by hparam columns, split_by). split_by: None = varied(df); [] = pool recipes (e.g. tuned)."""
    split_by = varied(df) if split_by is None else list(split_by)
    recipe_values = hparams(df)
    hparam_cols = [k for k in split_by if k in recipe_values]  # others (e.g. "recipe" from tag) already in df
    return df.drop(columns=hparam_cols, errors="ignore").join(recipe_values[hparam_cols]), split_by


def tag_by_grid(df: pd.DataFrame, grids: Mapping[str, str]) -> pd.DataFrame:
    """Rows whose config_hash is in grid {name: grid.yaml} -> recipe=name; in several -> duplicated; none
    (sweep, --set variants, old notebook rows) -> dropped."""
    from .grid import expand  # lazy: grid pulls models/torch

    parts = [df[df.config_hash.isin({run.run_id for run in expand(grid_path)})].assign(recipe=name)
             for name, grid_path in grids.items()]
    return pd.concat(parts, ignore_index=True)


def summary(df: pd.DataFrame, bench: pd.DataFrame | None = None,
            split_by: Sequence[str] | None = None) -> pd.DataFrame:
    """Mean ± std of top1_test over seeds per run key + split_by keys; + bench throughput if given."""
    df, split_by = with_varied(df, split_by)
    agg = (df.groupby(GROUP + split_by, dropna=False)
             .agg(top1=("top1_test", "mean"), std=("top1_test", "std"), n=("seed", "count"),
                  ft_mem_MB=("train_peak_mem_MB", "max"), ft_time_s=("train_time_s", "mean"))
             .reset_index())
    if bench is not None:
        agg = agg.merge(bench, on=["backbone", "in_chans"], how="left")
    return agg


def _delta(rows_a: pd.DataFrame, rows_b: pd.DataFrame, on: list[str], extra: Iterable[str] = ()) -> pd.DataFrame:
    """Seed-paired 100·(a − b) of top1_test -> mean/std/n per `on` (minus seed) + extra."""
    pairs = rows_a.merge(rows_b, on=on, suffixes=("_a", "_b"))  # NaN keys match NaN (notebook rows)
    pairs["delta"] = 100 * (pairs.top1_test_a - pairs.top1_test_b)
    group = [k for k in on if k != "seed"] + list(extra)
    return pairs.groupby(group, dropna=False).delta.agg(mean="mean", std="std", n="count").reset_index()


def paired_delta(df: pd.DataFrame, key: str, value_a: Any, value_b: Any,
                 split_by: Sequence[str] | None = None) -> pd.DataFrame:
    """pp delta rows[key]==value_a minus rows[key]==value_b, paired on everything else (frac+seed -> same subset)."""
    if split_by is not None and key not in df:  # key is an hparam -> needs its column
        split_by = [*split_by, key]
    df, split_by = with_varied(df, split_by)
    on = [k for k in [*KEY, *split_by] if k != key]
    rows_with = lambda value: df.loc[df[key] == value, on + ["top1_test"]]
    return _delta(rows_with(value_a), rows_with(value_b), on)


def ft_lp_gap(df: pd.DataFrame, split_by: Sequence[str] | None = None) -> pd.DataFrame:
    """pp FT − LP per seed; each FT recipe vs each LP recipe (shared split keys must match)."""
    df, split_by = with_varied(df, split_by)
    mode_only = set().union(*MODE_KEYS.values())
    on = [k for k in KEY if k != "mode"] + [k for k in split_by if k not in mode_only]  # shared + tags
    own_keys = {mode: [k for k in split_by if k in MODE_KEYS[mode]] for mode in MODE_KEYS}
    rows_of = lambda mode: df.loc[df["mode"] == mode, on + own_keys[mode] + ["top1_test"]]
    return _delta(rows_of("ft"), rows_of("lp"), on, extra=own_keys["ft"] + own_keys["lp"])


def table(agg: pd.DataFrame, split_by: Sequence[str] = ()) -> pd.DataFrame:
    """Wide: rows (backbone, init, *split_by), cols "FT 10%" = "mean ± std" (%); + bench cols if merged."""
    agg = agg.assign(**{k: agg[k].astype(object).where(agg[k].notna(), "–") for k in split_by})  # other mode's key
    std = agg["std"].fillna(0)
    cell = (100 * agg.top1).map("{:.1f}".format) + " ± " + (100 * std).map("{:.1f}".format)
    column = agg["mode"].str.upper() + " " + (100 * agg.frac).map("{:g}%".format)
    index_cols = ["backbone", "init", *split_by]
    wide = (agg.assign(cell=cell, column=column)
               .pivot_table(index=index_cols, columns="column", values="cell", aggfunc="first").fillna(""))
    bench_cols = [col for col in BENCH_COLS if col in agg]
    if bench_cols:
        wide = wide.join(agg.groupby(index_cols, dropna=False)[bench_cols].first().round(0))
    return wide


def pareto(points: pd.DataFrame, x: str = "infer_ips", y: str = "top1") -> pd.DataFrame:
    """Points no other point beats on both x and y (higher = better), sorted by x."""
    keep, best = [], float("-inf")
    for i, point in points.dropna(subset=[x, y]).sort_values([x, y], ascending=False).iterrows():
        if point[y] > best:
            keep.append(i)
            best = point[y]
    return points.loc[keep].sort_values(x)


def variants(agg: pd.DataFrame, split_by: list[str]) -> list[tuple[str, pd.DataFrame]]:
    """[(title, rows)]: one per FT combo of split_by values; NaN (other mode's key) matches any combo."""
    if not split_by:
        return [("", agg)]
    ft_rows = agg[agg["mode"] == "ft"] if (agg["mode"] == "ft").any() else agg
    out = []
    for combo in ft_rows[split_by].drop_duplicates().itertuples(index=False):
        mask = pd.Series(True, index=agg.index)
        for k, v in zip(split_by, combo):
            if pd.notna(v):
                mask &= agg[k].isna() | (agg[k] == v)
        title = ", ".join(f"{k}={v}" for k, v in zip(split_by, combo) if pd.notna(v))
        out.append((title, agg[mask]))
    return out


def plot_pareto(agg: pd.DataFrame, split_by: Sequence[str] = (), x: str = "infer_ips",
                xlabel: str = "inference img/s"):
    """Rows = split_by combos, cols = label fraction; legend/encoding in notebook md."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colour_of = dict(zip(sorted(agg.backbone.unique()), plt.cm.tab10.colors))
    marker_of = dict(zip(sorted(agg.init.unique()), "os^DvP"))
    fracs, groups = sorted(agg.frac.unique()), variants(agg, list(split_by))
    fig, axs = plt.subplots(len(groups), len(fracs), figsize=(5.5 * len(fracs), 4.5 * len(groups)),
                            sharex=True, squeeze=False)
    if agg[x].isna().any():
        log.warning("no bench row (skipped): %s",
                    agg.loc[agg[x].isna(), ["backbone", "in_chans"]].drop_duplicates().values.tolist())
    for ax_row, (title, rows) in zip(axs, groups):
        for ax, frac in zip(ax_row, fracs):
            points = rows[(rows.frac == frac) & rows[x].notna()]
            for _, point in points.iterrows():
                colour = colour_of[point.backbone]
                ax.errorbar(point[x], 100 * point.top1, yerr=100 * (point["std"] if pd.notna(point["std"]) else 0),
                            fmt=marker_of[point.init], ms=8, color=colour,
                            mfc=colour if point["mode"] == "ft" else "none", capsize=3)
            frontier = pareto(points[points["mode"] == "ft"], x)
            if len(frontier):
                ax.plot(frontier[x], 100 * frontier.top1, "k--", lw=1, zorder=0)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{100 * frac:g}% labels" + (f" | {title}" if title else ""))
        ax_row[0].set_ylabel("EuroSAT test top-1 (%)")
    handles = [Line2D([], [], color=colour, marker="o", ls="", label=backbone) for backbone, colour in colour_of.items()] + \
        [Line2D([], [], color="gray", marker=marker, ls="", label=init) for init, marker in marker_of.items()] + \
        [Line2D([], [], color="gray", marker="o", ls="", label="FT"),
         Line2D([], [], color="gray", marker="o", mfc="none", ls="", label="LP")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return fig
