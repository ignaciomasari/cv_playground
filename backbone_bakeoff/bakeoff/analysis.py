"""Results analysis: varied hparams (from row config JSON), seed-mean summary, paired deltas, Pareto."""
import json

import pandas as pd

from .config import MODE_KEYS
from .results import KEY

GROUP = [k for k in KEY if k != "seed"] + ["in_chans"]  # summary rows
BENCH_COLS = ["params_M", "infer_ips", "infer_mem_MB", "train_ips", "train_mem_MB"]  # bench.csv


def hparams(df):
    """Recipe values per row from `config` JSON (no config, e.g. notebook rows -> NaN). Run keys dropped."""
    cfgs = [json.loads(c) if isinstance(c, str) else {} for c in df.config]
    hp = pd.DataFrame(cfgs, index=df.index).drop(columns=KEY, errors="ignore")
    return hp.map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)  # hashable


def varied(df):
    """Recipe keys with >1 value within some (mode, frac) -> resolved epochs (set by frac) not counted."""
    hp = hparams(df)
    if hp.empty:
        return []
    n = hp.groupby([df["mode"], df.frac]).nunique().max()
    return [k for k in hp.columns if n[k] > 1]


def with_varied(df, var=None):
    """-> (df + varied hparam columns, their names). var: explicit keys (e.g. [] = pool tuned recipes)."""
    var = varied(df) if var is None else list(var)
    hp = hparams(df)
    cols = [k for k in var if k in hp]  # others (e.g. "recipe" from tag) already in df
    return df.drop(columns=cols, errors="ignore").join(hp[cols]), var


def tag_by_grid(df, grids):
    """Rows whose config_hash is in grid {name: grid.yaml} -> recipe=name; in several -> duplicated; none
    (sweep, --set variants, old notebook rows) -> dropped."""
    from .grid import expand  # lazy: grid pulls models/torch

    parts = [df[df.config_hash.isin({r.run_id for r in expand(p)})].assign(recipe=name)
             for name, p in grids.items()]
    return pd.concat(parts, ignore_index=True)


def summary(df, bench=None, var=None):
    """Mean ± std of top1_test over seeds per run key + varied hparams; + bench throughput if given."""
    df, var = with_varied(df, var)
    agg = (df.groupby(GROUP + var, dropna=False)
             .agg(top1=("top1_test", "mean"), std=("top1_test", "std"), n=("seed", "count"),
                  ft_mem_MB=("train_peak_mem_MB", "max"), ft_time_s=("train_time_s", "mean"))
             .reset_index())
    if bench is not None:
        agg = agg.merge(bench, on=["backbone", "in_chans"], how="left")
    return agg


def _delta(a, b, on, extra=()):
    """Seed-paired 100·(a − b) of top1_test -> mean/std/n per `on` (minus seed) + extra."""
    m = a.merge(b, on=on, suffixes=("_a", "_b"))  # NaN keys match NaN (notebook rows)
    m["d"] = 100 * (m.top1_test_a - m.top1_test_b)
    by = [k for k in on if k != "seed"] + list(extra)
    return m.groupby(by, dropna=False).d.agg(mean="mean", std="std", n="count").reset_index()


def paired_delta(df, col, a, b, var=None):
    """pp delta rows[col]==a minus rows[col]==b, paired on everything else (frac+seed -> same subset)."""
    if var is not None and col not in df:  # col is an hparam -> needs its column
        var = [*var, col]
    df, var = with_varied(df, var)
    on = [k for k in [*KEY, *var] if k != col]
    pick = lambda v: df.loc[df[col] == v, on + ["top1_test"]]
    return _delta(pick(a), pick(b), on)


def ft_lp_gap(df, var=None):
    """pp FT − LP per seed; each FT recipe vs each LP recipe (shared varied keys must match)."""
    df, var = with_varied(df, var)
    mode_only = set().union(*MODE_KEYS.values())
    on = [k for k in KEY if k != "mode"] + [k for k in var if k not in mode_only]  # shared + tags
    own = {m: [k for k in var if k in MODE_KEYS[m]] for m in MODE_KEYS}
    side = lambda m: df.loc[df["mode"] == m, on + own[m] + ["top1_test"]]
    return _delta(side("ft"), side("lp"), on, extra=own["ft"] + own["lp"])


def table(agg, var=()):
    """Wide: rows (backbone, init, *var), cols "FT 10%" = "mean ± std" (%); + bench cols if merged."""
    agg = agg.assign(**{k: agg[k].astype(object).where(agg[k].notna(), "–") for k in var})  # other mode's key
    std = agg["std"].fillna(0)
    cell = (100 * agg.top1).map("{:.1f}".format) + " ± " + (100 * std).map("{:.1f}".format)
    col = agg["mode"].str.upper() + " " + (100 * agg.frac).map("{:g}%".format)
    idx = ["backbone", "init", *var]
    t = agg.assign(cell=cell, col=col).pivot_table(index=idx, columns="col", values="cell", aggfunc="first").fillna("")
    eff = [c for c in BENCH_COLS if c in agg]
    if eff:
        t = t.join(agg.groupby(idx, dropna=False)[eff].first().round(0))
    return t


def pareto(d, x="infer_ips", y="top1"):
    """Points no other point beats on both x and y (higher = better), sorted by x."""
    keep, best = [], float("-inf")
    for i, r in d.dropna(subset=[x, y]).sort_values([x, y], ascending=False).iterrows():
        if r[y] > best:
            keep.append(i)
            best = r[y]
    return d.loc[keep].sort_values(x)


def variants(agg, var):
    """[(title, rows)]: one per FT combo of varied values; NaN (other mode's key) matches any combo."""
    if not var:
        return [("", agg)]
    src = agg[agg["mode"] == "ft"] if (agg["mode"] == "ft").any() else agg
    out = []
    for combo in src[var].drop_duplicates().itertuples(index=False):
        m = pd.Series(True, index=agg.index)
        for k, v in zip(var, combo):
            if pd.notna(v):
                m &= agg[k].isna() | (agg[k] == v)
        out.append((", ".join(f"{k}={v}" for k, v in zip(var, combo) if pd.notna(v)), agg[m]))
    return out


def plot_pareto(agg, var=(), x="infer_ips", xlabel="inference img/s"):
    """Rows = varied-hparam combos, cols = label fraction; legend/encoding in notebook md."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    col = dict(zip(sorted(agg.backbone.unique()), plt.cm.tab10.colors))
    mk = dict(zip(sorted(agg.init.unique()), "os^DvP"))
    fracs, groups = sorted(agg.frac.unique()), variants(agg, list(var))
    fig, axs = plt.subplots(len(groups), len(fracs), figsize=(5.5 * len(fracs), 4.5 * len(groups)),
                            sharex=True, squeeze=False)
    if agg[x].isna().any():
        print("no bench row (skipped):", agg.loc[agg[x].isna(), ["backbone", "in_chans"]].drop_duplicates().values.tolist())
    for row, (title, g) in zip(axs, groups):
        for ax, f in zip(row, fracs):
            d = g[(g.frac == f) & g[x].notna()]
            for _, r in d.iterrows():
                c = col[r.backbone]
                ax.errorbar(r[x], 100 * r.top1, yerr=100 * (r["std"] if pd.notna(r["std"]) else 0),
                            fmt=mk[r.init], ms=8, color=c, mfc=c if r["mode"] == "ft" else "none", capsize=3)
            fr = pareto(d[d["mode"] == "ft"], x)
            if len(fr):
                ax.plot(fr[x], 100 * fr.top1, "k--", lw=1, zorder=0)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{100 * f:g}% labels" + (f" | {title}" if title else ""))
        row[0].set_ylabel("EuroSAT test top-1 (%)")
    h = [Line2D([], [], color=c, marker="o", ls="", label=b) for b, c in col.items()] + \
        [Line2D([], [], color="gray", marker=m, ls="", label=i) for i, m in mk.items()] + \
        [Line2D([], [], color="gray", marker="o", ls="", label="FT"),
         Line2D([], [], color="gray", marker="o", mfc="none", ls="", label="LP")]
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return fig
