"""从 results/runs.jsonl 自动生成论文式报告：表格 + 图。

产物（都在 results/report/ 下）：
    REPORT.md                 主表 + 通信量表 + 无损性表
    main_table.csv            机器可读的主表
    fig_acc.png               每个数据集：各方法的集中 vs 联邦 ACC
    fig_metrics.png           四个指标：各方法（集中式）跨数据集对比
    fig_comm_vs_acc.png       联邦式：通信量 vs ACC
    fig_gap.png               集中式 vs 联邦式 ACC 差值（dumbbell）

用法：
    python scripts/make_report.py
    python scripts/make_report.py --log results/runs.jsonl --out results/report
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

METRICS = ("ACC", "NMI", "ARI", "Purity")
# 已知顺序只作为“优先排序”，新算法/新数据集会自动追加，不需要改这里。
KNOWN_DATASETS = ["Trento", "Augsburg", "Berlin"]
KNOWN_METHODS = ["osmagc", "etap", "samvgc", "example_kmeans"]
METHOD_LABEL = {
    "osmagc": "OSMAGC",
    "etap": "ETAP",
    "samvgc": "SAMVGC",
    "example_kmeans": "KMeans (baseline)",
}

# 参考调色板（light mode，已通过相邻对校验）
C_CENTRAL = "#2a78d6"   # slot 1 blue
C_FEDERATED = "#eb6834" # slot 2 orange
C_METHODS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#6250d6", "#e34948"]  # 最多 8 个方法
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"


def _ordered(items, known):
    """已知项按 known 的顺序在前，新项按字母序追加（方便以后加算法/数据集）。"""
    items = list(dict.fromkeys(items))
    return [x for x in known if x in items] + sorted(x for x in items if x not in known)


def _method_color(i: int) -> str:
    """不循环调色板：超过 8 个方法用中性灰，避免生成第 9 种颜色。"""
    return C_METHODS[i] if i < len(C_METHODS) else C_MUTED


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": C_SURFACE,
        "axes.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE,
        "font.size": 10,
        "axes.edgecolor": C_GRID,
        "axes.labelcolor": C_INK2,
        "xtick.color": C_MUTED,
        "ytick.color": C_MUTED,
        "grid.color": C_GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
    })
    return plt


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"找不到结果文件：{path}")
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
    return rows


def aggregate(rows: list[dict]) -> dict:
    """(dataset, method, partition) -> {metric: (mean, std, n), 'comm_bytes': float}"""
    buckets = defaultdict(lambda: defaultdict(list))
    comm = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["dataset"], r["method"], r["partition"])
        for m in METRICS:
            buckets[key][m].append(float(r[m]))
        cb = (r.get("extra") or {}).get("comm_bytes")
        if cb:
            comm[key] = float(cb)

    out = {}
    for key, md in buckets.items():
        rec = {}
        for m in METRICS:
            arr = np.asarray(md[m], dtype=float)
            rec[m] = (float(arr.mean()), float(arr.std(ddof=0)), int(arr.size))
        if key in comm:
            rec["comm_bytes"] = comm[key]
        out[key] = rec

    # 客户端数（同一 (dataset, method, partition) 应一致）
    for r in rows:
        if r.get("status") == "ok":
            key = (r["dataset"], r["method"], r["partition"])
            if key in out and "n_clients" not in out[key]:
                out[key]["n_clients"] = int(r.get("n_clients", 0))
    return out


def _fmt(rec, metric, with_std=True):
    if rec is None:
        return "—"
    mean, std, n = rec[metric]
    if with_std and n > 1:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def write_tables(agg: dict, out_dir: Path) -> None:
    lines = ["# 实验结果汇总", ""]
    lines.append("> 由 `scripts/make_report.py` 自动生成；"
                 "指标由 `common/metrics.py` 统一计算，只评 `gt>0` 的像素。")
    lines.append("")

    datasets = _ordered({k[0] for k in agg}, KNOWN_DATASETS)
    methods_all = _ordered({k[1] for k in agg}, KNOWN_METHODS)

    # 主表：每个数据集一张，行=方法，列=集中/联邦
    lines.append("## 一、主结果")
    lines.append("")
    for ds in datasets:
        methods = [m for m in methods_all if (ds, m, "none") in agg or (ds, m, "view") in agg]
        if not methods:
            continue
        lines.append(f"### {ds}")
        lines.append("")
        lines.append("| 方法 | 方式 | ACC | NMI | ARI | Purity | 通信量 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for m in methods:
            for part, pname in (("none", "集中式"), ("view", "联邦式")):
                rec = agg.get((ds, m, part))
                if rec is None:
                    continue
                cb = rec.get("comm_bytes")
                cbs = f"{cb/1e6:.1f} MB" if cb else "—"
                lines.append(
                    f"| {METHOD_LABEL.get(m, m)} | {pname} | "
                    f"{_fmt(rec,'ACC')} | {_fmt(rec,'NMI')} | "
                    f"{_fmt(rec,'ARI')} | {_fmt(rec,'Purity')} | {cbs} |"
                )
        lines.append("")

    # 通信量表
    lines.append("## 二、联邦式通信量")
    lines.append("")
    lines.append("| 数据集 | 方法 | 客户端数 | 通信量 |")
    lines.append("|---|---|---:|---:|")
    for ds in datasets:
        for m in methods_all:
            rec = agg.get((ds, m, "view"))
            if rec is None or "comm_bytes" not in rec:
                continue
            ncl = rec.get("n_clients", "—")
            lines.append(f"| {ds} | {METHOD_LABEL.get(m, m)} | {ncl} | {rec['comm_bytes']/1e6:.1f} MB |")
    lines.append("")

    # 无损性：自动扫描 results/verify_*.jsonl（新增方法的验证脚本不用改这里）
    verify_lines = []
    for p in sorted(out_dir.parent.glob("verify_*.jsonl")):
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("status") != "ok":
                continue
            e = r.get("extra") or {}
            rounds = e.get("comm_rounds", "—")
            verify_lines.append(
                f"| {r['dataset']} | {r['method']} | {rounds} | "
                f"{e.get('lossless_max_dH', '—')} | "
                f"{e.get('central_ACC', float('nan')):.4f} | {r['ACC']:.4f} |"
            )
    if verify_lines:
        lines.append("## 三、联邦化无损性验证")
        lines.append("")
        lines.append("| 数据集 | 方法 | 轮数 | max\\|ΔH\\| | 集中式 ACC | 联邦式 ACC |")
        lines.append("|---|---|---:|---:|---:|---:|")
        lines.extend(verify_lines)
        lines.append("")
        lines.append("> 轮数小于默认值（10）的记录是冒烟验证，只用于确认无损性；"
                     "其 ACC 不与主表比较。")
        lines.append("")

    # 图
    lines.append("## 四、图")
    lines.append("")
    lines.append("### 各方法集中式 vs 联邦式 ACC")
    lines.append("")
    lines.append("![ACC](fig_acc.png)")
    lines.append("")
    lines.append("### 四个指标（集中式）")
    lines.append("")
    lines.append("![metrics](fig_metrics.png)")
    lines.append("")
    lines.append("### 联邦式：通信量 vs ACC（对数横轴）")
    lines.append("")
    lines.append("![comm](fig_comm_vs_acc.png)")
    lines.append("")
    lines.append("### 集中式 vs 联邦式 ACC 差值")
    lines.append("")
    lines.append("![gap](fig_gap.png)")
    lines.append("")

    # 像素图：如果 results/maps 下有 PNG，就嵌进来
    maps_dir = out_dir.parent / "maps"
    map_pngs = sorted(maps_dir.glob("*.png")) if maps_dir.exists() else []
    if map_pngs:
        lines.append("## 五、像素图（GT vs 预测）")
        lines.append("")
        lines.append("> 由 `run_all.py --save-maps` + `scripts/plot_maps.py` 生成。")
        lines.append("")
        for png in map_pngs:
            rel = f"../maps/{png.name}"
            lines.append(f"### {png.stem}")
            lines.append("")
            lines.append(f"![{png.stem}]({rel})")
            lines.append("")

    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    # CSV
    csv_lines = ["dataset,method,partition,ACC,NMI,ARI,Purity,comm_bytes,n_seeds"]
    for (ds, m, part), rec in sorted(agg.items()):
        csv_lines.append(
            f"{ds},{m},{part},"
            f"{rec['ACC'][0]:.6f},{rec['NMI'][0]:.6f},{rec['ARI'][0]:.6f},{rec['Purity'][0]:.6f},"
            f"{int(rec.get('comm_bytes', 0))},{rec['ACC'][2]}"
        )
    (out_dir / "main_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")


def plot_acc(agg: dict, out_dir: Path) -> None:
    plt = _setup_matplotlib()
    datasets = _ordered({k[0] for k in agg}, KNOWN_DATASETS)
    methods_all = _ordered({k[1] for k in agg}, KNOWN_METHODS)
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.6), sharex=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        methods = [m for m in methods_all if (ds, m, "none") in agg or (ds, m, "view") in agg]
        y = np.arange(len(methods))
        h = 0.36
        for offset, part, color, label in ((-h/2 - 0.02, "none", C_CENTRAL, "集中式"),
                                           (h/2 + 0.02, "view", C_FEDERATED, "联邦式")):
            vals, errs, present = [], [], []
            for m in methods:
                rec = agg.get((ds, m, part))
                vals.append(rec["ACC"][0] if rec else np.nan)
                errs.append(rec["ACC"][1] if rec else 0.0)
                present.append(rec is not None)
            vals = np.asarray(vals)
            errs = np.asarray(errs)
            ax.barh(y + offset, vals, height=h, color=color, label=label,
                    xerr=errs if np.any(errs > 0) else None,
                    error_kw=dict(ecolor=C_MUTED, lw=1, capsize=2))
            for yi, v, ok in zip(y + offset, vals, present):
                if ok and np.isfinite(v):
                    ax.text(v + 0.015, yi, f"{v:.3f}", va="center", ha="left",
                            fontsize=7.5, color=C_INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABEL.get(m, m) for m in methods], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.08)
        ax.set_xlabel("ACC")
        ax.set_title(ds, fontsize=11, color=C_INK, pad=8)
        ax.grid(axis="y", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "fig_acc.png", dpi=200)
    plt.close(fig)


def plot_metrics(agg: dict, out_dir: Path) -> None:
    plt = _setup_matplotlib()
    datasets = _ordered({k[0] for k in agg}, KNOWN_DATASETS)
    methods = _ordered({k[1] for k in agg if k[2] == "none"}, KNOWN_METHODS)
    if not datasets or not methods:
        return
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4), sharey=False)
    x = np.arange(len(datasets))
    width = 0.8 / max(len(methods), 1)
    for ax, metric in zip(axes, METRICS):
        for i, m in enumerate(methods):
            vals, errs = [], []
            for d in datasets:
                rec = agg.get((d, m, "none"))
                vals.append(rec[metric][0] if rec else np.nan)
                errs.append(rec[metric][1] if rec else 0.0)
            vals = np.asarray(vals); errs = np.asarray(errs)
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width * 0.9,
                   color=_method_color(i),
                   label=METHOD_LABEL.get(m, m),
                   yerr=errs if np.any(errs > 0) else None,
                   error_kw=dict(ecolor=C_MUTED, lw=1, capsize=2))
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=8)
        ax.set_title(metric, fontsize=10, color=C_INK)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), frameon=False,
               bbox_to_anchor=(0.5, 1.03), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "fig_metrics.png", dpi=200)
    plt.close(fig)


def plot_comm_vs_acc(agg: dict, out_dir: Path) -> None:
    plt = _setup_matplotlib()
    points = []
    for (ds, m, part), rec in agg.items():
        if part != "view" or "comm_bytes" not in rec:
            continue
        points.append((ds, m, rec["comm_bytes"] / 1e6, rec["ACC"][0]))
    if not points:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    methods = _ordered({p[1] for p in points}, KNOWN_METHODS)
    for i, m in enumerate(methods):
        pts = [p for p in points if p[1] == m]
        if not pts:
            continue
        xs = [p[2] for p in pts]
        ys = [p[3] for p in pts]
        ax.scatter(xs, ys, s=42, color=_method_color(i),
                   edgecolor=C_SURFACE, linewidth=1.2, zorder=3,
                   label=METHOD_LABEL.get(m, m))
        for ds, _, x, y in pts:
            ax.annotate(ds[:3], (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5, color=C_INK2)
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter
    ticks = [100, 200, 500, 1000, 2000, 5000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("通信量 / MB（对数轴）")
    ax.set_ylabel("ACC")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="both", color=C_GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_comm_vs_acc.png", dpi=200)
    plt.close(fig)


def plot_gap(agg: dict, out_dir: Path) -> None:
    """ΔACC = 联邦式 − 集中式。点图，x 轴围绕 0，便于看“是否无损”。"""
    plt = _setup_matplotlib()
    pairs = []
    for (ds, m, part), rec_c in agg.items():
        if part != "none":
            continue
        rec_f = agg.get((ds, m, "view"))
        if rec_f is None:
            continue
        pairs.append((ds, m, rec_f["ACC"][0] - rec_c["ACC"][0]))
    if not pairs:
        return
    datasets = _ordered({p[0] for p in pairs}, KNOWN_DATASETS)
    methods = _ordered({p[1] for p in pairs}, KNOWN_METHODS)
    pairs.sort(key=lambda p: (datasets.index(p[0]), methods.index(p[1])))
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(pairs) + 1.4))
    y = np.arange(len(pairs))
    for yi, (ds, m, d) in zip(y, pairs):
        color = C_FEDERATED if abs(d) > 1e-4 else C_MUTED
        ax.scatter([d], [yi], s=52, color=color, edgecolor=C_SURFACE,
                   linewidth=1.2, zorder=3)
        ax.text(d, yi - 0.18, f"{d:+.4f}", ha="center", va="bottom",
                fontsize=7.5, color=C_INK2)
    ax.axvline(0, color=C_MUTED, lw=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{METHOD_LABEL.get(m, m)} · {ds}" for ds, m, _ in pairs], fontsize=8.5)
    ax.invert_yaxis()
    lim = max(0.004, max(abs(d) for _, _, d in pairs) * 1.35)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("ΔACC = 联邦式 − 集中式")
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_gap.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "results" / "runs.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "report"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(Path(args.log))
    agg = aggregate(rows)
    if not agg:
        print("没有可用的成功结果。")
        return 1

    write_tables(agg, out_dir)
    plot_acc(agg, out_dir)
    plot_metrics(agg, out_dir)
    plot_comm_vs_acc(agg, out_dir)
    plot_gap(agg, out_dir)

    print(f"报告已生成：{out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
