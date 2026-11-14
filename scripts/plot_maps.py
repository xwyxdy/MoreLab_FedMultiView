"""把 run_all.py --save-maps 存下的标签图渲染成像素图（GT vs 预测）。

用法：
    # 先让 run_all 存标签图（--force 强制重跑，--save-maps 存 .npy）
    python scripts/run_all.py --datasets Trento --methods osmagc --seeds 0 \
        --save-maps --force --log results/aux/maps_run.jsonl

    # 再画图
    python scripts/plot_maps.py --dataset Trento --methods osmagc etap samvgc

输出：results/maps/<dataset>_<method>_<partition>_seed<seed>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_dataset  # noqa: E402
from common.metrics import acc as acc_fn  # noqa: E402

# 参考调色板：0=背景灰，1..8 用前 8 个分类色（像素图仅用于可视化）
CLASS_COLORS = [
    "#e1e0d9",  # 0 背景
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
]
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"


def _align_to_gt(gt_flat: np.ndarray, pred: np.ndarray, mask: np.ndarray):
    """用有标注像素上的匈牙利匹配，把预测标签映射到真值类别（仅用于可视化）。"""
    from scipy.optimize import linear_sum_assignment

    t = gt_flat[mask]
    p = pred[mask]
    true_labels = np.unique(t)
    pred_labels = np.unique(p)
    t_idx = np.searchsorted(true_labels, t)
    p_idx = np.searchsorted(pred_labels, p)
    C = np.bincount(p_idx * len(true_labels) + t_idx,
                    minlength=len(pred_labels) * len(true_labels)).reshape(
        len(pred_labels), len(true_labels))
    row, col = linear_sum_assignment(-C)
    mapping = {int(pred_labels[r]): int(true_labels[c]) for r, c in zip(row, col)}

    mapped = np.zeros_like(pred, dtype=np.int32)
    for pl, tl in mapping.items():
        mapped[pred == pl] = tl
    return mapped, mapping


def _plot_one(ds, method: str, pred: np.ndarray, out_path: Path,
              partition: str, seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": C_SURFACE,
        "savefig.facecolor": C_SURFACE,
        "text.color": C_INK,
    })

    H, W = ds.H, ds.W
    gt_flat = ds.flatten_gt("F")
    mask = gt_flat > 0
    mapped, _ = _align_to_gt(gt_flat, pred, mask)
    acc = acc_fn(gt_flat[mask], mapped[mask])

    gt_2d = gt_flat.reshape(H, W, order="F")
    pred_2d = mapped.reshape(H, W, order="F")

    n_cls = ds.n_classes
    cmap = ListedColormap(CLASS_COLORS[: n_cls + 1])
    norm = BoundaryNorm(np.arange(-0.5, n_cls + 1.5, 1), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, 13 * H / W * 0.5)))
    for ax, img, title in zip(
        axes,
        (gt_2d, pred_2d),
        ("Ground Truth", f"{method} ({partition})  ACC={acc:.4f}"),
    ):
        ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
        ax.set_title(title, fontsize=11, color=C_INK, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # 图例：类别编号 1..K
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[k]) for k in range(1, n_cls + 1)]
    fig.legend(handles, [f"class {k}" for k in range(1, n_cls + 1)],
               loc="lower center", ncol=min(n_cls, 8), frameon=False, fontsize=8.5)
    fig.suptitle(f"{ds.name} · {method} · {partition} · seed={seed}",
                 fontsize=12, color=C_INK, y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Trento")
    ap.add_argument("--methods", nargs="+", default=["osmagc"])
    ap.add_argument("--partition", default="none", choices=["none", "view"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--maps-dir", default=str(ROOT / "results" / "maps"))
    ap.add_argument("--out", default=None, help="输出目录，默认与 maps-dir 相同")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    maps_dir = Path(args.maps_dir)
    out_dir = Path(args.out) if args.out else maps_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    for method in args.methods:
        npy = maps_dir / f"{ds.name}__{method}__{args.partition}__seed{args.seed}.npy"
        if not npy.exists():
            print(f"  [跳过] {method}: 找不到 {npy.name}（先用 run_all --save-maps --force 生成）")
            continue
        pred = np.load(npy).reshape(-1).astype(np.int32)
        if pred.size != ds.n_pixels:
            print(f"  [跳过] {method}: 标签数 {pred.size} != {ds.n_pixels}")
            continue
        out_png = out_dir / f"{ds.name}__{method}__{args.partition}__seed{args.seed}.png"
        acc = _plot_one(ds, method, pred, out_png, args.partition, args.seed)
        print(f"  ✓ {method}: ACC={acc:.4f} -> {out_png}")
        found += 1

    if found == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
