"""统一跑批入口：把 methods/ 下的算法，在各数据集上跑一遍，结果追加到 results/runs.jsonl。

用法
----
    python scripts/run_all.py                          # 全部数据集 × 全部算法 × 默认种子
    python scripts/run_all.py --datasets Trento
    python scripts/run_all.py --methods my_algo --seeds 0 1 2
    python scripts/run_all.py --list                   # 只列出发现了哪些算法

自动发现
--------
扫 `methods/*.py`，凡是以 `_` 开头的文件跳过（模板 `_template.py` 会被跳过），
其余文件里**只要定义了 `run_<文件名>(ds, seed=...)`** 就会被当成一个算法。

结果落盘
--------
每条记录写一行 JSON 到 `results/runs.jsonl`，含：
  - 协议指纹（数据集 / 划分 / 种子 / BLAS 线程数）
  - 四个指标（ACC / NMI / ARI / Purity）
  - 耗时、状态


**跑完一组就立刻落盘**，中途被杀不丢已完成的。
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_dataset                       # noqa: E402
from common.metrics import evaluate                        # noqa: E402
from common.partition import view_partition                # noqa: E402
from common.protocol import RunResult, append_result, done_keys  # noqa: E402
from common.threads import THREADS                         # noqa: E402


def discover_methods() -> dict:
    """扫 methods/ 找可用算法，返回 {算法名: 可调用对象}。"""
    out = {}
    mdir = ROOT / "methods"
    for f in sorted(mdir.glob("*.py")):
        if f.name.startswith("_"):
            continue                                    # 模板与内部文件跳过
        name = f.stem
        try:
            mod = importlib.import_module(f"methods.{name}")
        except Exception as exc:                        # noqa: BLE001
            print(f"  [跳过] {name}: 导入失败 {type(exc).__name__}: {exc}")
            continue
        fn = getattr(mod, f"run_{name}", None)
        if callable(fn):
            out[name] = fn
        else:
            print(f"  [跳过] {name}: 没找到 run_{name}() 函数（见 methods/_template.py）")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Augsburg", "Trento", "Berlin"],
                    help="要跑哪些数据集；默认全部三个 "
                         "（Augsburg Trento Berlin）")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="只跑指定算法；默认跑 methods/ 下全部发现到的")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="随机种子；默认只跑 0。要多种子写 --seeds 0 1 2")
    ap.add_argument("--log", default=str(ROOT / "results" / "runs.jsonl"))
    ap.add_argument("--list", action="store_true", help="只列出发现的算法")
    args = ap.parse_args()

    methods = discover_methods()
    print("=" * 88)
    print(f"已发现的算法（{len(methods)} 个）: {list(methods) or '（无，见 methods/_template.py）'}")
    print(f"BLAS 线程数 = {THREADS}（已固定，勿改；详见 docs/协议与规范.md）")
    print("=" * 88)
    if args.list:
        return 0
    if not methods:
        print("\n没有发现任何算法。照 methods/_template.py 写一个，再跑本脚本。")
        return 0

    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}
        if not methods:
            print(f"\n--methods 指定的算法都不存在")
            return 1

    log = Path(args.log)
    done = done_keys(log)                    # 断点续跑：已完成的跳过

    for dname in args.datasets:
        ds = load_dataset(dname)
        print(f"\n--- {ds.summary()} ---")
        for seed in args.seeds:
            for mname, fn in methods.items():
                key = (ds.name, mname, "none", "zscore", seed)
                if key in done:
                    print(f"    {mname:<22} 已完成，跳过")
                    continue
                try:
                    t0 = __import__("time").time()
                    labels, detail = fn(ds, seed=seed)
                    el = __import__("time").time() - t0
                    gt = ds.flatten_gt("F")
                    mask = gt > 0
                    m = evaluate(gt[mask], labels[mask])
                    print(f"    {mname:<22} ACC={m['ACC']:.4f} NMI={m['NMI']:.4f} "
                          f"ARI={m['ARI']:.4f}  ({el:.0f}s)", flush=True)
                    append_result(RunResult(
                        dataset=ds.name, method=mname, partition="none",
                        partition_mode="view", n_clients=1, seed=seed,
                        ACC=float(m["ACC"]), NMI=float(m["NMI"]),
                        ARI=float(m["ARI"]), Purity=float(m["Purity"]),
                        runtime_s=el, n_clusters=int(ds.n_classes),
                        extra={k: v for k, v in (detail or {}).items()
                               if isinstance(v, (int, float, str, bool))},
                    ), log)
                except Exception as exc:                 # noqa: BLE001
                    print(f"    {mname:<22} 失败: {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    append_result(RunResult(
                        dataset=ds.name, method=mname, partition="none",
                        partition_mode="view", n_clients=1, seed=seed,
                        status="error", note=f"{type(exc).__name__}: {exc}",
                    ), log)

    print(f"\n结果已追加到 {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
