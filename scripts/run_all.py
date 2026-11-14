"""统一跑批入口：把 methods/ 下的算法，在各数据集上跑一遍，结果追加到 results/runs.jsonl。

用法
----
    python scripts/run_all.py                          # 全部数据集 × 全部算法 × 默认种子
    python scripts/run_all.py --datasets Trento
    python scripts/run_all.py --methods my_algo --seeds 0 1 2
    python scripts/run_all.py --federated              # 除集中式外，再跑各方法的联邦式
    python scripts/run_all.py --list                   # 只列出发现了哪些算法

自动发现
--------
扫 `methods/*.py`，凡是以 `_` 开头的文件跳过（模板 `_template.py` 会被跳过），
其余文件里：
  - `run_<文件名>(ds, seed=...)`            → 集中式入口
  - `run_<文件名>_federated(ds, part, ...)` → 联邦式入口（视图划分，可选）
两者至少有一个即被当作一个算法。

结果落盘
--------
每条记录写一行 JSON 到 `results/runs.jsonl`，含：
  - 协议指纹（数据集 / 划分 / 种子 / BLAS 线程数 / norm）
  - 四个指标（ACC / NMI / ARI / Purity）
  - 耗时、状态、通信量（联邦式）

**跑完一组就立刻落盘**，中途被杀不丢已完成的。
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_dataset                       # noqa: E402
from common.metrics import evaluate                        # noqa: E402
from common.partition import view_partition                # noqa: E402
from common.protocol import RunResult, append_result, done_keys  # noqa: E402
from common.threads import THREADS                         # noqa: E402


@dataclass
class MethodSpec:
    name: str
    central: Callable | None
    federated: Callable | None
    norm: str


def discover_methods() -> dict[str, MethodSpec]:
    """扫 methods/ 找可用算法，返回 {算法名: MethodSpec}。"""
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
        central = getattr(mod, f"run_{name}", None)
        federated = getattr(mod, f"run_{name}_federated", None)
        if not callable(central) and not callable(federated):
            print(f"  [跳过] {name}: 没找到 run_{name} 或 run_{name}_federated（见 methods/_template.py）")
            continue
        out[name] = MethodSpec(
            name=name,
            central=central if callable(central) else None,
            federated=federated if callable(federated) else None,
            norm=getattr(mod, "NORM", "zscore"),
        )
    return out


def _jsonable(v, depth: int = 0):
    """把 detail 里能进 JSON 的值保留，其余（ndarray/自定义对象）丢弃。"""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return None                                     # H_list 等大数组绝不落盘
    if isinstance(v, (list, tuple)) and depth < 3:
        return [_jsonable(x, depth + 1) for x in v]
    if isinstance(v, dict) and depth < 3:
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    return None                                         # 自定义对象丢弃


def _clean_detail(detail) -> dict:
    out = {}
    for k, v in (detail or {}).items():
        j = _jsonable(v)
        if j is not None:
            out[k] = j
    return out


def main() -> int:
    # ══════════════════════════════════════════════════════════════════════
    # 跑批主流程
    #
    #   ① 解析参数
    #   ② discover_methods()  扫描 methods/，找到 run_<名> 和 run_<名>_federated
    #   ③ 逐数据集 load_dataset()  读 .mat → RSDataset（F 序展平）
    #   ④ 逐 seed × 逐方法：
    #        集中式: fn(ds, seed=seed)              → (labels, detail)
    #        联邦式: fn(ds, part, seed=seed)        → (labels, detail)
    #   ⑤ evaluate()  只在 gt>0 的像素上算 ACC/NMI/ARI/Purity
    #   ⑥ append_result()  一条 JSON 追加进 results/runs.jsonl（断点续跑靠它）
    #
    # 关键：方法只负责“算出每像素一个标签”，评分和落盘**全部由本脚本统一做**，
    #       所以不同方法的数字才可比（见 docs/协议与规范.md 第一节“四条统一”）。
    # ══════════════════════════════════════════════════════════════════════
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["Augsburg", "Trento", "Berlin"],
                    help="要跑哪些数据集；默认全部三个（Augsburg Trento Berlin）")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="只跑指定算法；默认跑 methods/ 下全部发现到的")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="随机种子；默认只跑 0。要多种子写 --seeds 0 1 2")
    ap.add_argument("--federated", action="store_true",
                    help="除集中式外，再跑各方法的联邦式（视图划分）")
    ap.add_argument("--save-maps", action="store_true",
                    help="把每次运行的标签图存到 results/maps/*.npy（用于画像素图）")
    ap.add_argument("--force", action="store_true",
                    help="忽略断点续跑，强制重跑（配合 --save-maps 使用）")
    ap.add_argument("--log", default=str(ROOT / "results" / "runs.jsonl"))
    ap.add_argument("--list", action="store_true", help="只列出发现的算法")
    args = ap.parse_args()

    methods = discover_methods()
    print("=" * 88)
    print(f"已发现的算法（{len(methods)} 个）: {list(methods)}")
    print(f"BLAS 线程数 = {THREADS}（已固定，勿改；详见 docs/协议与规范.md）")
    if args.federated:
        print("联邦模式：除集中式外，还会跑 run_<名>_federated（视图划分）")
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
    # 读已有结果 → 得到一个“已完成”集合。去重键 = (数据集, 方法, 划分, norm, 种子)。
    # 这就是“中断后重跑会自动跳过已完成”的实现方式。
    # --force 时清空集合，强制重跑（画像素图时需要重新拿到 labels）。
    done = set() if args.force else done_keys(log)
    if args.force and log.resolve() == (ROOT / "results" / "runs.jsonl").resolve():
        print("[提示] --force 会向主结果 runs.jsonl 追加重复记录；"
              "建议改用 --log results/aux/force_run.jsonl")

    maps_dir = ROOT / "results" / "maps"
    if args.save_maps:
        maps_dir.mkdir(parents=True, exist_ok=True)

    for dname in args.datasets:              # ③ 逐数据集
        ds = load_dataset(dname)             # 数据层：见 common/data.py
        print(f"\n--- {ds.summary()} ---")
        # 联邦划分只构造一次：视图划分 = 每个视图一个客户端（common/partition.py）
        part = view_partition(ds) if args.federated else None

        for seed in args.seeds:              # ④ 逐种子
            for name, spec in methods.items():   # ④ 逐方法
                # 一个方法可能有两个入口：集中式和联邦式。
                # 这里把“要跑的任务”列成一个 list，下面统一执行，避免两份重复代码。
                jobs = []
                if spec.central is not None:
                    # (类型, 函数, 传入的 part, 记录用 partition, mode, 客户端数)
                    jobs.append(("central", spec.central, None,
                                 "none", "none", 1))
                if args.federated and spec.federated is not None:
                    jobs.append(("federated", spec.federated, part,
                                 part.shape, part.mode, len(part)))

                for kind, fn, part_arg, pname, pmode, ncl in jobs:
                    # 去重键：同一组协议只跑一次。
                    # norm 也在键里 —— 因为 OSMAGC/ETAP/SAMVGC 的预处理不是 z-score。
                    key = (ds.name, name, pname, spec.norm, seed)
                    if key in done:
                        print(f"    {name:<22} ({kind:<9}) 已完成，跳过")
                        continue
                    try:
                        t0 = time.time()
                        # ══════════════════════════════════════════════
                        # 调用算法。约定：返回 (labels, detail)
                        #   labels: (N,) int32，每像素一个簇标签，F 序
                        #   detail: dict，方法自己报告耗时/超参/通信量
                        # 联邦式多传一个 part（Partition 对象）
                        # ══════════════════════════════════════════════
                        if part_arg is None:
                            labels, detail = fn(ds, seed=seed) #集中
                        else:
                            labels, detail = fn(ds, part_arg, seed=seed) #联邦
                        el = time.time() - t0
                        labels = np.asarray(labels).reshape(-1)

                        if labels.size != ds.n_pixels:
                            raise ValueError(f"标签数 {labels.size} != 像素数 {ds.n_pixels}")
                        # 可选：保存标签图（F 序的 (N,) int32），供 scripts/plot_maps.py 画像素图
                        if args.save_maps:
                            map_path = maps_dir / f"{ds.name}__{name}__{pname}__seed{seed}.npy"
                            np.save(map_path, labels.astype(np.int32))
                        # ⑤ 统一评分：只在有标注像素（gt>0）上算四个指标。
                        # 聚类本身用了全部像素（含背景），评分时把背景排除。
                        gt = ds.flatten_gt("F")
                        mask = gt > 0
                        m = evaluate(gt[mask], labels[mask])
                        # detail 里可能有 ndarray / 自定义对象，先清成 JSON-safe
                        extra = _clean_detail(detail)
                        if kind == "federated" and "comm_bytes" not in extra:
                            print(f"    {name:<22} [警告] 联邦方法未在 detail 里提供 comm_bytes")
                        comm = extra.get("comm_bytes")
                        comm_s = f"  comm={comm/1e6:.1f}MB" if comm else ""
                        print(f"    {name:<22} ({kind:<9}) ACC={m['ACC']:.4f} "
                              f"NMI={m['NMI']:.4f} ARI={m['ARI']:.4f} "
                              f"PUR={m['Purity']:.4f} ({el:.0f}s){comm_s}", flush=True)
                        append_result(RunResult(
                            dataset=ds.name, method=name, partition=pname,
                            partition_mode=pmode, n_clients=ncl, norm=spec.norm,
                            seed=seed,
                            ACC=float(m["ACC"]), NMI=float(m["NMI"]),
                            ARI=float(m["ARI"]), Purity=float(m["Purity"]),
                            runtime_s=el, n_clusters=int(ds.n_classes),
                            extra=extra,
                        ), log)
                    except Exception as exc:                 # noqa: BLE001
                        print(f"    {name:<22} ({kind:<9}) 失败: {type(exc).__name__}: {exc}")
                        traceback.print_exc()
                        append_result(RunResult(
                            dataset=ds.name, method=name, partition=pname,
                            partition_mode=pmode, n_clients=ncl, norm=spec.norm,
                            seed=seed, status="error",
                            note=f"{type(exc).__name__}: {exc}",
                        ), log)

    print(f"\n结果已追加到 {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
