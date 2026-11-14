"""SAMVGC 适配器（集中式，后续研究联邦化）。

论文：Structure-Adaptive Multi-View Graph Clustering for Remote Sensing Data（AAAI 2025）。

核心见 `vendor/samvgc/core.py`（PyTorch）。预处理是逐视图 minmax + 融合 SLIC，
协议指纹记 `norm="samvgc-specific"`。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data import RSDataset  # noqa: E402

NORM = "samvgc-specific"


@dataclass(frozen=True)
class SAMVGCParams:
    n_superpixels: int = 2000      # 超像素/锚点数 M
    k_neighbors: int = 10          # 近邻数 k
    central_rate: float = 0.5      # 中心样本比例 α
    epochs: int = 100              # 训练轮数
    lr: float = 1e-3               # 学习率
    lam1: float = 100.0            # 对比损失权重 λ1
    lam2: float = 500.0            # 一致性损失权重 λ2
    refine_every: int = 10         # 图精炼间隔 ip


# 三个数据集用同一套默认超参（论文未给，取惯例值）。
DEFAULT_PARAMS: dict[str, SAMVGCParams] = {
    "Trento":   SAMVGCParams(),
    "Augsburg": SAMVGCParams(),
    "Berlin":   SAMVGCParams(n_superpixels=4000),   # 规模大，锚点稍多
}


def params_for(ds: RSDataset) -> SAMVGCParams:
    if ds.name not in DEFAULT_PARAMS:
        raise KeyError(f"SAMVGC 没有 {ds.name!r} 的超参；可用：{sorted(DEFAULT_PARAMS)}")
    return DEFAULT_PARAMS[ds.name]


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run_samvgc(ds: RSDataset, seed: int = 0, params: SAMVGCParams | None = None,
               verbose: bool = False):
    """SAMVGC 集中式。返回 (labels, detail)。"""
    from vendor.samvgc.core import run_samvgc as _run_samvgc

    # ① 取本数据集超参
    params = params or params_for(ds)
    t0 = time.time()

    # ② 交给 vendor/samvgc/core.py 跑完整流程（PyTorch）：
    #    融合视图 → SLIC 超像素锚点 → 逐视图 kNN 图 → 3 层 GCN 编解码
    #    → 循环 100 轮 { 重建损失 + 融合 Z_F → K-means → 中心/混淆样本
    #                     → 原型对比损失 + 一致性损失 → 反向传播 }
    #    → 每 10 轮图精炼 → 最终 K-means → 锚点标签传播到像素
    y_pred = _run_samvgc(
        cubes=list(ds.cubes),
        n_classes=ds.n_classes,
        n_superpixels=params.n_superpixels,
        k_neighbors=params.k_neighbors,
        central_rate=params.central_rate,
        epochs=params.epochs,
        lr=params.lr,
        lam1=params.lam1,
        lam2=params.lam2,
        refine_every=params.refine_every,
        device=_device(),
        seed=seed,
        verbose=verbose,
    )

    labels = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    if labels.size != ds.n_pixels:
        raise ValueError(f"标签数 {labels.size} != 像素数 {ds.n_pixels}")

    detail = {
        "runtime_s": float(time.time() - t0),
        "norm": NORM,
        "n_superpixels": int(params.n_superpixels),
        "k_neighbors": int(params.k_neighbors),
        "central_rate": float(params.central_rate),
        "epochs": int(params.epochs),
        "lr": float(params.lr),
        "lam1": float(params.lam1),
        "lam2": float(params.lam2),
        "device": _device(),
    }
    return labels, detail


def run_samvgc_federated(ds: RSDataset, part, seed: int = 0,
                         params: SAMVGCParams | None = None, verbose: bool = False):
    """SAMVGC 联邦化版本（视图划分）。返回 (labels, detail)，detail 含 comm_bytes。"""
    from vendor.samvgc.core import run_samvgc as _run_samvgc

    # ══════════════════════════════════════════════════════════════════════
    # 联邦执行流程（详见 docs/SAMVGC联邦化实验报告.md）：
    #   客户端 v：本地 GCN 编码/解码 + 图精炼 + 重建损失
    #   服务器  ：接收各视图 Z(v) → 融合 Z_F = Σ Wv ⊙ Z(v)
    #             → K-means 簇分配 → 融合原型 µF_k → 广播
    #   客户端  ：用广播的簇结构算对比损失 + 一致性损失，更新本地 GCN
    # SLIC 超像素当公开预处理（一次性，不构成每轮交流）。
    # 注意：这是深度方法，100 epoch 每轮都交换特征，通信量比确定性方法大一个量级。
    # ══════════════════════════════════════════════════════════════════════
    params = params or params_for(ds)
    t0 = time.time()

    y_pred, comm = _run_samvgc(
        cubes=list(ds.cubes),
        n_classes=ds.n_classes,
        n_superpixels=params.n_superpixels,
        k_neighbors=params.k_neighbors,
        central_rate=params.central_rate,
        epochs=params.epochs,
        lr=params.lr,
        lam1=params.lam1,
        lam2=params.lam2,
        refine_every=params.refine_every,
        device=_device(),
        seed=seed,
        verbose=verbose,
        return_comm=True,
    )

    labels = np.asarray(y_pred, dtype=np.int32).reshape(-1)
    if labels.size != ds.n_pixels:
        raise ValueError(f"标签数 {labels.size} != 像素数 {ds.n_pixels}")

    detail = {
        "runtime_s": float(time.time() - t0),
        "norm": NORM,
        "partition": getattr(part, "shape", "view"),
        "n_clients": int(ds.n_views),
        "n_superpixels": int(params.n_superpixels),
        "k_neighbors": int(params.k_neighbors),
        "central_rate": float(params.central_rate),
        "epochs": int(params.epochs),
        "lr": float(params.lr),
        "lam1": float(params.lam1),
        "lam2": float(params.lam2),
        "device": _device(),
        "comm_bytes": int(comm["comm_bytes"]),
        "comm_up_bytes": int(comm["comm_up_bytes"]),
        "comm_down_bytes": int(comm["comm_down_bytes"]),
        "comm_rounds": int(comm["comm_rounds"]),
        "comm_per_client_per_round": int(comm["comm_per_client_per_round"]),
        "n_anchors": int(comm["n_anchors"]),
        "d_hat": int(comm["d_hat"]),
        "comm_accounting": "float32; 每轮每客户端: 上传 Z(v)(M×d̂), "
                           "下发 ZF(M×d̂)+原型(K×d̂)+簇标签(M)",
    }
    return labels, detail
