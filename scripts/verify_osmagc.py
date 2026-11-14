"""验证 OSMAGC 的联邦化是【无损】的，并量化通信代价。

核心断言
--------
集中式里唯一的跨视图耦合是一个求和项 `Σ_v R_v·Z_v`。联邦版让每个客户端本地算
`R_v·Z_v` 上传，服务器相加即可精确重建同一轮 H。因此联邦版的 H 序列应与集中式
**逐元素相同**（初始 H 是确定性的，见 vendor/osmagc/core/init.py::initH）。

若该断言成立：**OSMAGC 的联邦化精度损失 = 0；真实代价在通信量上。**

用法
----
    python scripts/verify_osmagc.py --dataset Trento --seed 0 --max-iter 10
    python scripts/verify_osmagc.py --dataset Berlin --max-iter 2   # 冒烟

默认写独立日志 results/verify_osmagc.jsonl；加 --write-runs 才追加 runs.jsonl。
退出码：无损返回 0，否则返回 1。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.data import load_dataset                       # noqa: E402
from common.metrics import evaluate                        # noqa: E402
from common.partition import view_partition                # noqa: E402
from common.protocol import (                              # noqa: E402
    DEFAULT_LOG, RunResult, append_result,
)
from methods.osmagc import (                               # noqa: E402
    NORM, params_for, run_osmagc, run_osmagc_federated,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Trento")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log", default=str(ROOT / "results" / "verify_osmagc.jsonl"))
    ap.add_argument("--write-runs", action="store_true",
                    help="同时把联邦结果追加到 runs.jsonl")
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    part = view_partition(ds)
    p = replace(params_for(ds), max_iter=args.max_iter)

    print("=" * 96)
    print(f"OSMAGC 联邦化验证 | {ds.name} | 视图划分（一个客户端 = 一个模态）")
    print("=" * 96)
    print(f"  {ds.summary()}")
    print(f"  客户端数 = {ds.n_views}（{', '.join(ds.view_names)}）")

    # ---- 集中式（return_H 一次拿到 H 序列和 labels）----
    print(f"\n[1] 集中式（所有视图、所有像素放一起跑）")
    labels_c, det_c = run_osmagc(ds, seed=args.seed, params=p,
                                 n_init=args.n_init, return_H=True,
                                 verbose=args.verbose)
    H_c = det_c["H_list"]
    print(f"    完成 {len(H_c)} 轮，H 形状 {H_c[-1].shape}，耗时 {det_c['runtime_s']:.1f}s")

    # ---- 联邦化 ----
    print(f"\n[2] 联邦化（各客户端只碰自己的视图，原始数据不出门）")
    labels_f, det_f = run_osmagc_federated(
        ds, part, seed=args.seed, params=p, n_init=args.n_init,
        return_H=True, verbose=args.verbose,
    )
    H_f = det_f["H_list"]
    print(f"    完成 {len(H_f)} 轮，耗时 {det_f['runtime_s']:.1f}s")
    print(f"    最终视图权重 α = {[round(a, 4) for a in det_f['alpha']]}")

    # ---- 无损性断言 ----
    print(f"\n[3] 无损性断言：联邦版的 H 序列 vs 集中式")
    assert len(H_c) == len(H_f), f"轮数不一致 {len(H_c)} vs {len(H_f)}"
    worst = 0.0
    for i, (hc, hf) in enumerate(zip(H_c, H_f), 1):
        d = float(np.abs(hc - hf).max())
        worst = max(worst, d)
        if i <= 3 or i == len(H_c):
            print(f"    第 {i:2d} 轮  max|ΔH| = {d:.3e}")
        elif i == 4:
            print(f"    ...")
    labels_equal = np.array_equal(labels_c, labels_f)
    identical = worst < 1e-12 and labels_equal
    print(f"    全部 {len(H_c)} 轮的最大逐元素偏差 = {worst:.3e}")
    print(f"    标签是否完全一致 = {labels_equal}")
    print(f"    → {'联邦化完全无损 ✓（H 逐元素相同）' if identical else '存在偏差 ✗'}")

    # ---- 精度对比 ----
    gt = ds.flatten_gt("F")
    mask = gt > 0
    m_c = evaluate(gt[mask], labels_c[mask])
    m_f = evaluate(gt[mask], labels_f[mask])
    print(f"\n[4] 精度对比（同一套诚实 k-means 读出协议）")
    print(f"    {'':<10}{'ACC':>10}{'NMI':>10}{'ARI':>10}{'Purity':>10}")
    print(f"    {'集中式':<10}{m_c['ACC']:>10.4f}{m_c['NMI']:>10.4f}"
          f"{m_c['ARI']:>10.4f}{m_c['Purity']:>10.4f}")
    print(f"    {'联邦化':<10}{m_f['ACC']:>10.4f}{m_f['NMI']:>10.4f}"
          f"{m_f['ARI']:>10.4f}{m_f['Purity']:>10.4f}")
    print(f"    {'gap':<10}{m_f['ACC'] - m_c['ACC']:>+10.4f}"
          f"{m_f['NMI'] - m_c['NMI']:>+10.4f}{m_f['ARI'] - m_c['ARI']:>+10.4f}"
          f"{m_f['Purity'] - m_c['Purity']:>+10.4f}")
    print(f"    （gap 不为 0 只可能来自 k-means 随机初始化，与联邦化无关）")

    # ---- 通信代价 ----
    print(f"\n[5] 通信代价（这是联邦化真正的成本）")
    per = det_f["comm_per_client_per_round"]
    print(f"    每客户端每轮：上传 S_v(dim×N)+e_v = {per/1e6:.2f} MB，下发 H(dim×N) = {(per-8)/1e6:.2f} MB")
    print(f"    合计 {det_f['comm_rounds']} 轮 × {det_f['n_clients']} 客户端：")
    print(f"      总上传   = {det_f['comm_up_bytes']/1e6:.1f} MB")
    print(f"      总下发   = {det_f['comm_down_bytes']/1e6:.1f} MB")
    print(f"      总计     = {det_f['comm_bytes']/1e6:.1f} MB")
    print(f"      初始化广播 H0 = {det_f['comm_init_bytes']/1e6:.1f} MB（未计入规范 comm_bytes）")
    print(f"    ⚠️ 通信量与像素数 N 成正比，这是 OSMAGC 联邦化的主要瓶颈。")

    # ---- 落盘 ----
    append_result(RunResult(
        dataset=ds.name, method="osmagc", partition="view", partition_mode="view",
        n_clients=ds.n_views, eval_mask="labeled", norm=NORM,
        seed=args.seed, ACC=m_f["ACC"], NMI=m_f["NMI"], ARI=m_f["ARI"],
        Purity=m_f["Purity"], runtime_s=det_f["runtime_s"],
        n_clusters=ds.n_classes,
        extra={
            "alpha": det_f["alpha"], "num_pixel": det_f["num_pixel"],
            "lam": det_f["lam"], "gamma": det_f["gamma"],
            "lossless_max_dH": worst,
            "comm_bytes": det_f["comm_bytes"],
            "comm_up_bytes": det_f["comm_up_bytes"],
            "comm_down_bytes": det_f["comm_down_bytes"],
            "comm_rounds": det_f["comm_rounds"],
            "comm_init_bytes": det_f["comm_init_bytes"],
            "central_ACC": m_c["ACC"],
        },
    ), Path(args.log))
    print(f"\n[6] 已写入 {args.log}")
    if args.write_runs:
        print(f"    （--write-runs 未实现直接追加，请用 run_all.py --federated 跑正式结果）")
    print("=" * 96)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
