"""验证 ETAP 的联邦化是【无损】的，并量化通信代价。

核心断言
--------
ETAP 的跨视图耦合全部在锚点级（m×m）：
  - α_v / β_v：标量聚合
  - F：加权拉普拉斯和 + 特征分解（服务器）
  - 张量 Schatten-p：所有视图 S_v 堆成张量做 t-SVD（服务器）
  - 最终共识：S / Z 的加权平均（服务器）

联邦版把跨视图步骤标成服务器、其余逐视图本地算，计算本身与集中式逐函数复用，
因此标签**逐位相同**。

用法
----
    python scripts/verify_etap.py --dataset Trento --seed 0 --max-iter 10
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
from common.protocol import RunResult, append_result       # noqa: E402
from methods.etap import NORM, params_for, run_etap, run_etap_federated  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Trento")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--log", default=str(ROOT / "results" / "verify_etap.jsonl"))
    args = ap.parse_args()

    ds = load_dataset(args.dataset)
    part = view_partition(ds)
    p = replace(params_for(ds), iter_num=args.max_iter)

    print("=" * 96)
    print(f"ETAP 联邦化验证 | {ds.name} | 视图划分（一个客户端 = 一个模态）")
    print("=" * 96)
    print(f"  {ds.summary()}")
    print(f"  客户端数 = {ds.n_views}（{', '.join(ds.view_names)}）")

    labels_c, det_c = run_etap(ds, params=p)
    print(f"\n[1] 集中式 完成，耗时 {det_c['runtime_s']:.1f}s")

    labels_f, det_f = run_etap_federated(ds, part, params=p)
    print(f"[2] 联邦化 完成，耗时 {det_f['runtime_s']:.1f}s")

    identical = np.array_equal(labels_c, labels_f)
    print(f"\n[3] 无损性断言：标签是否逐位一致 = {identical}")
    print(f"    → {'联邦化完全无损 ✓' if identical else '存在偏差 ✗'}")

    gt = ds.flatten_gt("F")
    mask = gt > 0
    m_c = evaluate(gt[mask], labels_c[mask])
    m_f = evaluate(gt[mask], labels_f[mask])
    print(f"\n[4] 精度对比")
    print(f"    {'':<10}{'ACC':>10}{'NMI':>10}{'ARI':>10}{'Purity':>10}")
    print(f"    {'集中式':<10}{m_c['ACC']:>10.4f}{m_c['NMI']:>10.4f}"
          f"{m_c['ARI']:>10.4f}{m_c['Purity']:>10.4f}")
    print(f"    {'联邦化':<10}{m_f['ACC']:>10.4f}{m_f['NMI']:>10.4f}"
          f"{m_f['ARI']:>10.4f}{m_f['Purity']:>10.4f}")

    print(f"\n[5] 通信代价")
    print(f"    每客户端每轮（锚点级 m×m）: {det_f['comm_per_client_per_round']/1e3:.1f} KB")
    print(f"    每轮合计（{det_f['comm_rounds']} 轮）: "
          f"{(det_f['comm_bytes'] - det_f['comm_final_z_bytes'])/1e6:.2f} MB")
    print(f"    最终 Z_v 上传（N×m，一次性）: {det_f['comm_final_z_bytes']/1e6:.2f} MB")
    print(f"    总计上传 = {det_f['comm_up_bytes']/1e6:.1f} MB, "
          f"下发 = {det_f['comm_down_bytes']/1e6:.1f} MB, "
          f"合计 = {det_f['comm_bytes']/1e6:.1f} MB")

    # 落盘：与 verify_osmagc.jsonl 同格式，make_report.py 会自动扫描
    append_result(RunResult(
        dataset=ds.name, method="etap", partition="view", partition_mode="view",
        n_clients=ds.n_views, eval_mask="labeled", norm=NORM,
        seed=args.seed, ACC=m_f["ACC"], NMI=m_f["NMI"], ARI=m_f["ARI"],
        Purity=m_f["Purity"], runtime_s=det_f["runtime_s"],
        n_clusters=ds.n_classes,
        extra={
            "lossless_max_dH": 0.0 if identical else float("nan"),
            "labels_identical": bool(identical),
            "comm_bytes": det_f["comm_bytes"],
            "comm_up_bytes": det_f["comm_up_bytes"],
            "comm_down_bytes": det_f["comm_down_bytes"],
            "comm_rounds": det_f["comm_rounds"],
            "comm_init_bytes": 0,
            "central_ACC": m_c["ACC"],
        },
    ), Path(args.log))
    print(f"    已写入 {args.log}")

    print("=" * 96)
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
