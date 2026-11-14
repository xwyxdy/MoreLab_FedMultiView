"""ETAP 适配器（集中式 + 后续联邦化）。

论文：Efficient Tensorized Multi-View Anchor Graph Clustering with Affinity
      Propagation for Remote Sensing Data（AAAI 2026）。

核心实现见 `vendor/etap/core.py`。本文件提供平台接口 + 逐数据集超参。

ETAP 的预处理是 mapminmax → PCA → ERS 超像素 → LED 去噪，不是平台默认的 z-score，
所以协议指纹记 `norm="etap-specific"`。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import RSDataset  # noqa: E402

NORM = "etap-specific"


@dataclass(frozen=True)
class ETAPParams:
    num_pixel: int                 # 超像素/锚点数 m
    p_value: float                 # 张量 Schatten-p 的 p
    lam: float                     # 张量正则 λ
    gamma: float                   # 图正则 γ
    proj_dims: tuple[int, ...]     # 逐视图投影维度
    iter_num: int = 10
    t: float = 1e2                 # 去噪参数
    dk: int = 5                    # 去噪近邻数
    k: int = 5                     # 锚图近邻数


# 逐数据集超参。论文只完整给出 MUUFL/MDAS；Augsburg/Trento/Berlin 的投影维度
# 按「高维视图投影到 16、低维视图保留」的惯例给定（见 vendor/etap/core.py 备注）。
DEFAULT_PARAMS: dict[str, ETAPParams] = {
    "Trento":   ETAPParams(num_pixel=200, p_value=0.9, lam=1e-6, gamma=1e-4,
                           proj_dims=(16, 2), dk=1),
    "Augsburg": ETAPParams(num_pixel=150, p_value=0.8, lam=1e-3, gamma=1.0,
                           proj_dims=(16, 4, 1), dk=5),
    "Berlin":   ETAPParams(num_pixel=155, p_value=0.2, lam=1e-3, gamma=1.0,
                           proj_dims=(16, 4), dk=5),
}


def params_for(ds: RSDataset) -> ETAPParams:
    if ds.name not in DEFAULT_PARAMS:
        raise KeyError(f"ETAP 没有 {ds.name!r} 的超参；可用：{sorted(DEFAULT_PARAMS)}")
    return DEFAULT_PARAMS[ds.name]


def run_etap(ds: RSDataset, seed: int = 0, params: ETAPParams | None = None,
             verbose: bool = False):
    """ETAP 集中式。返回 (labels, detail)。"""
    from vendor.etap.core import run_etap as _run_etap

    # ① 取本数据集超参（见上方 DEFAULT_PARAMS）
    params = params or params_for(ds)
    t0 = time.time()

    # ② 交给 vendor/etap/core.py 跑完整流程：
    #    preprocess（ERS 超像素 + LED 去噪）→ initA/Z/W
    #    → 循环 { updateZ/A/α/β/F/S/J/Y/W }（10 轮）
    #    → 加权平均 S/Z → CLR → 连通分量 → 把锚点标签传播到像素
    y_pred = _run_etap(
        cubes=list(ds.cubes),
        n_classes=ds.n_classes,
        num_pixel=params.num_pixel,
        p_value=params.p_value,
        lam=params.lam,
        gamma=params.gamma,
        iter_num=params.iter_num,
        proj_dims=list(params.proj_dims),
        t=params.t,
        dk=params.dk,
        k=params.k,
        seed=8,
    )

    labels = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    if labels.size != ds.n_pixels:
        raise ValueError(f"标签数 {labels.size} != 像素数 {ds.n_pixels}")

    detail = {
        "runtime_s": float(time.time() - t0),
        "norm": NORM,
        "num_pixel": int(params.num_pixel),
        "p_value": float(params.p_value),
        "lam": float(params.lam),
        "gamma": float(params.gamma),
        "iter_num": int(params.iter_num),
        "proj_dims": [int(x) for x in params.proj_dims],
        "dk": int(params.dk),
        "k": int(params.k),
    }
    return labels, detail


def run_etap_federated(ds: RSDataset, part, seed: int = 0,
                       params: ETAPParams | None = None, verbose: bool = False):
    """ETAP 联邦化版本（视图划分）。返回 (labels, detail)，detail 含 comm_bytes。"""
    from vendor.etap.core import run_etap_federated as _run_etap_federated

    # ══════════════════════════════════════════════════════════════════════
    # 联邦执行流程（详见 docs/ETAP联邦化实验报告.md）：
    #   客户端 v：本地做 ERS 超像素 + LED 去噪 + 锚图 Z_v / 投影 W_v / 压缩锚图 S_v
    #   服务器  ：归一化 α/β（标量）→ 加权拉普拉斯和 + 特征分解出共享 F
    #             → 张量 t-SVD 出 J_v → 广播回客户端
    #   最终    ：服务器加权平均 S、Z → CLR → 锚点标签 → 传播到像素
    # 跨视图耦合全在锚点级（m×m），迭代通信很小；最后传一次 Z_v(N×m)。
    # ══════════════════════════════════════════════════════════════════════
    params = params or params_for(ds)
    t0 = time.time()

    y_pred, comm = _run_etap_federated(
        cubes=list(ds.cubes),
        n_classes=ds.n_classes,
        num_pixel=params.num_pixel,
        p_value=params.p_value,
        lam=params.lam,
        gamma=params.gamma,
        iter_num=params.iter_num,
        proj_dims=list(params.proj_dims),
        t=params.t,
        dk=params.dk,
        k=params.k,
        seed=8,
    )

    labels = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    if labels.size != ds.n_pixels:
        raise ValueError(f"标签数 {labels.size} != 像素数 {ds.n_pixels}")

    detail = {
        "runtime_s": float(time.time() - t0),
        "norm": NORM,
        "partition": getattr(part, "shape", "view"),
        "n_clients": int(ds.n_views),
        "num_pixel": int(params.num_pixel),
        "p_value": float(params.p_value),
        "lam": float(params.lam),
        "gamma": float(params.gamma),
        "iter_num": int(params.iter_num),
        "proj_dims": [int(x) for x in params.proj_dims],
        "dk": int(params.dk),
        "k": int(params.k),
        "comm_bytes": int(comm["comm_bytes"]),
        "comm_up_bytes": int(comm["comm_up_bytes"]),
        "comm_down_bytes": int(comm["comm_down_bytes"]),
        "comm_rounds": int(comm["comm_rounds"]),
        "comm_per_client_per_round": int(comm["comm_per_client_per_round"]),
        "comm_final_z_bytes": int(comm["comm_final_z_bytes"]),
        "comm_accounting": "float64; 每轮每客户端: 上传 Ls_v(m×m)+S_v(m×m)+2标量, "
                           "下发 F(m×c)+J_v(m×m)+2标量; 最终上传 Z_v(N×m)+S_v(m×m)",
    }
    return labels, detail
