"""OSMAGC 集中式 + 联邦化（视图划分）适配器。

================================================================================
一、OSMAGC 是什么
================================================================================
Orthogonal Spatial-Aware Multi-View Anchor Graph Clustering（CVPR 2026），
面向不完整多视图遥感数据的锚图聚类，四个模块：

    SAGI   空间感知锚点初始化（PCA + ERS 超像素 → 锚点 → 初始锚图 Z）
    MSAGL  多尺度锚图学习（逐视图矩阵分解，X_v ≈ A_v Z_v）
    SACFL  结构对齐共识特征学习（R_v 把各视图锚图对齐到共享 H）
    OSAR   正交空间感知正则（水平/垂直 TV，用公开栅格几何 D 平滑 H）

主循环（Algorithm 1）：A → R → Z → H → α，共 max_iter 轮，最后一轮 H 做 k-means。

================================================================================
二、联邦化改了什么（视图划分 / 纵向联邦）
================================================================================
集中式五步里，A_v / R_v / Z_v 本来就**逐视图独立**，唯一跨视图耦合是
H 更新里的一个求和项：

        H  ←  solve( λ · Σ_v R_v Z_v )

于是把计算按视图拆到客户端、把求和搬到服务器：

    客户端 v（持有 X_v、q_v，原始数据不出门）
        A_v ← updateA(X_v, Z_v)                      本地
        R_v ← updateRv(Z_v, H)                       本地（H 来自服务器）
        Z_v ← updateZv(A_v, X_v, R_v, H, α_v)        本地
        上传 S_v = R_v @ Z_v（dim×N）和标量残差 e_v
    服务器（持有 H、α，以及公开栅格几何 D）
        H   ← solve( λ · Σ_v S_v )                   只做求和 + 一次稀疏求解
        α_v ← e_v / Σ e_v                            标量归一化
        下发 H、α_v

    Σ_v S_v 是精确求和，且各视图 A/R/Z 更新逐位独立，因此只要初始化、参数、
    迭代次序、BLAS 线程数、视图求和顺序一致，联邦版与集中式的 H 序列**逐元素
    相同**（10 轮 max|ΔH| = 0），最终聚类指标完全一致。联邦化只改变数据与计算
    的部署方式，代价是通信量（与像素数 N 成正比），不改变精度。

通信量口径（每客户端每轮，float64）：
    上传 S_v（dim×N）+ e_v（8B）；下发 H（dim×N）。
    初始化广播 H0（dim×N）单列另计，不混入规范字段 comm_bytes。
================================================================================
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

# OSMAGC 的预处理不是平台默认的 zscore（内部是 mapminmax → PCA → ERS → 逐像素 L2），
# 所以协议指纹里的 norm 必须单独记为 osmagc-specific，否则与其他方法的数字不可比。
NORM = "osmagc-specific"


# --------------------------------------------------------------------------- #
# 超参
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OSMAGCParams:
    num_pixel: tuple[int, ...]     # 逐视图超像素/锚点数
    lam: float
    gamma: float
    max_iter: int = 10
    # 复现验证时确认必须的开关（见 OSMAGC 项目 NOTES.md D1/D10/D13）
    h_simplex: bool = False        # D1: 官方该步是死代码，开着 H 会塌缩（NMI≈0）
    mapminmax_axis: int = 0        # D13: 0=按列(按波段)
    h_denom_v: bool = True         # D10: True=跟随论文(分母带 V)
    sagi_mode: str = "ers"
    sacfl_fixed: bool = False


# 各数据集的实测最优超参（旧平台 tuning 结论）。协议指纹一部分，勿改。
DEFAULT_PARAMS: dict[str, OSMAGCParams] = {
    "Augsburg": OSMAGCParams(num_pixel=(200, 200, 200), lam=1.0, gamma=10.0),
    "Trento":   OSMAGCParams(num_pixel=(100, 100),      lam=1.0, gamma=100.0),
    "Berlin":   OSMAGCParams(num_pixel=(100, 100),      lam=1e-3, gamma=10.0),
}


def params_for(ds: RSDataset) -> OSMAGCParams:
    if ds.name not in DEFAULT_PARAMS:
        raise KeyError(
            f"OSMAGC 没有 {ds.name!r} 的调好超参；可用：{sorted(DEFAULT_PARAMS)}"
        )
    return DEFAULT_PARAMS[ds.name]


# --------------------------------------------------------------------------- #
# 数据转换
# --------------------------------------------------------------------------- #
def _to_osmagc_dataset(ds: RSDataset):
    """把平台 RSDataset 转成 OSMAGC 的 Dataset。

    两者展平顺序都是 Fortran（列优先），像素索引一一对应。
    三个数据集都是完整数据（无像素级缺失），index 全为 arange(N)。
    """
    from vendor.osmagc.data.loader import Dataset as OSMAGCDataset

    N = ds.n_pixels
    gt_flat = ds.flatten_gt("F")
    index = (
        [np.arange(N, dtype=np.int64) for _ in range(ds.n_views)]
        if ds.index is None
        else [np.asarray(ix, dtype=np.int64) for ix in ds.index]
    )
    rate = float(np.mean([ds.missing_rate(v) for v in range(ds.n_views)]))
    return OSMAGCDataset(
        name=ds.name,
        cubes=list(ds.cubes),
        gt=gt_flat,
        index=index,
        rate=rate,
        H=ds.H,
        W=ds.W,
        V=ds.n_views,
    )


def _client_setup(ds: RSDataset, params: OSMAGCParams, v: int) -> dict:
    """客户端 v 的本地准备：只碰自己的那个视图（SAGI + 锚点初始化）。"""
    from vendor.osmagc.core.init import initA
    from vendor.osmagc.core.predata import predata
    from vendor.osmagc.data.loader import findindex

    ods_data = [ds.cubes[v].reshape((-1, ds.cubes[v].shape[2]), order="F")
                .T.astype(np.float64)]
    index_v = ([np.arange(ds.n_pixels, dtype=np.int64)] if ds.index is None
               else [np.asarray(ds.index[v], dtype=np.int64)])

    X_v, Q_v = findindex(ods_data, index_v)          # [(Bv,N)], (N,1)
    q_v = Q_v[:, 0].astype(np.float64)

    pre_v = predata([ds.cubes[v]], [int(params.num_pixel[v])],
                    pca_sign=1, mapminmax_axis=params.mapminmax_axis,
                    sagi_mode=params.sagi_mode)
    A_v = initA(X_v, pre_v.sp_labels, [int(params.num_pixel[v])])[0]

    return {
        "X": X_v[0],                                 # (Bv, N) L2 归一化
        "q": q_v,                                    # (N,)
        "A": A_v,                                    # (Bv, Mv)
        "sp_labels": pre_v.sp_labels[0],
        "unm": int(q_v.sum()),
    }


def _honest_readout(H_last: np.ndarray, K: int, seed: int, n_init: int) -> np.ndarray:
    """诚实读出：最后一轮 H + KMeans(n_init, random_state=seed)，只看 inertia。

    绝不使用真值挑轮次或挑初始化（见 docs/协议与规范.md 第三节）。
    """
    from sklearn.cluster import KMeans

    X = np.ascontiguousarray(H_last.T)               # (N, dim)
    km = KMeans(n_clusters=K, n_init=n_init, random_state=seed)
    return km.fit_predict(X).astype(np.int32)


# --------------------------------------------------------------------------- #
# 集中式
# --------------------------------------------------------------------------- #
def run_osmagc(
    ds: RSDataset,
    seed: int = 0,
    params: OSMAGCParams | None = None,
    n_init: int = 10,
    verbose: bool = False,
    return_H: bool = False,
):
    """OSMAGC 集中式：所有视图、所有像素放一起跑。

    Returns
    -------
    (labels, detail)
        labels : (N,) int32，全体像素簇标签（诚实读出，无 oracle 选择）
        detail : 元信息；return_H=True 时额外带 H_list（供无损性验证）
    """
    from vendor.osmagc.core.model import OSMAGCConfig, run_osmagc as _run_osmagc

    # ① 取这个数据集的超参（逐数据集调过，属于协议指纹）
    params = params or params_for(ds)
    # ② 组装 OSMAGC 官方配置。开关值见 vendor/osmagc/core/model.py 注释：
    #    h_simplex=False  官方那步是死代码，开着 H 会塌缩（NOTES.md D1）
    #    h_denom_v=True   跟随论文 Eq.(11)
    cfg = OSMAGCConfig(
        num_pixel=list(params.num_pixel),
        lam=params.lam,
        gamma=params.gamma,
        dim=None,                       # None → 取类别数 c
        max_iter=params.max_iter,
        h_simplex=params.h_simplex,
        mapminmax_axis=params.mapminmax_axis,
        h_denom_v=params.h_denom_v,
        sagi_mode=params.sagi_mode,
        sacfl_fixed=params.sacfl_fixed,
        verbose=verbose,
    )

    t0 = time.time()
    # ③ 数据转换：RSDataset → OSMAGC Dataset（字段一一对应，F 序一致）
    ods = _to_osmagc_dataset(ds)
    # ④ 跑集中式主循环 A→R→Z→H→α，返回每轮的 H 序列
    res = _run_osmagc(ods, cfg)
    runtime = time.time() - t0

    # ⑤ 诚实读出：只取最后一轮 H，KMeans(n_init=10, random_state=seed)
    labels = _honest_readout(res.H_list[-1], ds.n_classes, seed, n_init)

    detail = {
        "runtime_s": float(runtime),
        "norm": NORM,
        "alpha": [float(x) for x in res.alpha],
        "num_pixel": [int(x) for x in params.num_pixel],
        "lam": float(params.lam),
        "gamma": float(params.gamma),
        "max_iter": int(params.max_iter),
        "n_rounds": int(len(res.H_list)),
        "dim": int(res.dim),
    }
    if return_H:
        detail["H_list"] = res.H_list
    return labels, detail


# --------------------------------------------------------------------------- #
# 联邦式（视图划分）
# --------------------------------------------------------------------------- #
def _check_view_partition(ds: RSDataset, part) -> dict:
    """校验划分是合法的视图划分，返回 view→client 的映射。

    要求：每个客户端恰好持有一个视图，视图编号覆盖 0..V-1 且无重复。
    """
    from common.partition import Partition

    if not isinstance(part, Partition) or part.mode != "view":
        raise ValueError("OSMAGC 联邦化只支持视图划分（partition.mode == 'view'）")
    if len(part.clients) != ds.n_views:
        raise ValueError(
            f"视图划分的客户端数 {len(part.clients)} != 视图数 {ds.n_views}"
        )
    client_of_view = {}
    for c in part.clients:
        if len(c.view_idx) != 1:
            raise ValueError(f"客户端 {c.client_id} 应恰好持有 1 个视图，实际 {c.view_idx}")
        v = int(c.view_idx[0])
        if v < 0 or v >= ds.n_views:
            raise ValueError(f"客户端 {c.client_id} 的视图编号 {v} 越界")
        if v in client_of_view:
            raise ValueError(f"视图 {v} 被多个客户端持有")
        client_of_view[v] = c.client_id
    return client_of_view


def run_osmagc_federated(
    ds: RSDataset,
    part,
    seed: int = 0,
    params: OSMAGCParams | None = None,
    n_init: int = 10,
    verbose: bool = False,
    return_H: bool = False,
):
    """OSMAGC 联邦化版本（视图划分，无损）。

    Returns
    -------
    (labels, detail)
        labels : (N,) int32，与集中式同口径的诚实读出
        detail : 含 comm_bytes 等通信量字段；return_H=True 时额外带 H_list
    """
    from vendor.osmagc.core.init import initZ, initH
    from vendor.osmagc.core.operators import construct_D
    from vendor.osmagc.core.updates import (
        HUpdater, updateA, updateRv, updateZv,
    )

    params = params or params_for(ds)
    _check_view_partition(ds, part)

    V, N = ds.n_views, ds.n_pixels
    dim = ds.n_classes
    lam, gamma = params.lam, params.gamma

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════
    # 联邦执行流程（对照集中式五步 A→R→Z→H→α）：
    #   客户端 v：本地 SAGI/initA/initZ → 每轮本地 updateA/updateRv/updateZv
    #             → 上传 S_v = R_v Z_v 和标量 e_v
    #   服务器  ：Σ_v S_v → 解 H → 归一化 α → 下发 H, α
    # 数学与集中式完全相同，只是把“求和”从内存搬到网络。
    # ══════════════════════════════════════════════════════════════════════

    # ---------- 客户端本地准备（各自只碰自己的视图）----------
    # _client_setup 内部：视图立方体 → L2 归一化 X_v、可用掩码 q_v、
    # ERS 超像素 → 锚点 A_v。原始数据只在这里被读一次。
    clients = [_client_setup(ds, params, v) for v in range(V)]

    # 初始锚图：initZ 需要各视图可用像素，逐客户端算（不通信）
    idx_list = ([np.arange(N, dtype=np.int64)] * V if ds.index is None
                else [np.asarray(ds.index[v], dtype=np.int64) for v in range(V)])
    unm = np.array([c["unm"] for c in clients])
    Z = initZ([c["X"][:, c["q"] == 1] for c in clients], [c["A"] for c in clients],
              dim, idx_list, unm, N)

    # ---------- 服务器：公开栅格几何 + 共享状态 ----------
    # D 只由 (H,W) 决定，不含任何视图数据 —— 相当于“公开的图结构”。
    D = construct_D(ds.H, ds.W)                       # 公开：所有客户端同图
    # HUpdater 内部对 (λI + γD) 做一次稀疏 LU 分解，之后每轮复用（Berlin 能跑动的关键）
    solver = HUpdater(D, lam, gamma, V, use_v_factor=params.h_denom_v)
    H = initH(N, dim)                                 # 确定性初始化（所有客户端一致）
    alpha = np.ones(V) / V                            # 初始视图权重均等

    # ---------- 联邦主循环 ----------
    H_list = []
    bytes_mat = 8 * dim * N                           # 一个 (dim,N) float64 矩阵的字节数
    comm_up = 0
    comm_down = 0
    comm_rounds = 0

    for it in range(1, params.max_iter + 1):
        S_sum = np.zeros((dim, N), dtype=np.float64)
        e = np.zeros(V, dtype=np.float64)

        # --- 各客户端本地迭代（互相不通信）---
        # 对应集中式的 A → R → Z 三步，全部只依赖本视图 + 服务器下发的 H/α。
        # ⚠️ 必须按 v=0..V-1 顺序累加 S_sum，与集中式 updateH 的求和顺序一致，
        #    否则浮点末位可能不同，破坏“逐元素相同”的严格断言。
        for v in range(V):
            Xv, qv = clients[v]["X"], clients[v]["q"]
            Av = updateA([Xv], [Z[v]], [qv])[0]        # ① 锚点（正交 Procrustes）
            Rv = updateRv(1, [Z[v]], H)[0]             # ② 对齐矩阵（SVD）
            Zv = updateZv([Av], [Xv], N, [Rv], H, lam, # ③ 锚图（单形投影）
                          np.array([alpha[v]]), [qv])[0]

            clients[v]["A"] = Av
            Z[v] = Zv

            S_v = Rv @ Zv                               # 本视图对全局 H 的贡献 (dim,N)
            avail = qv == 1
            e[v] = float(np.linalg.norm(               # 本视图残差（用于 α）
                Xv[:, avail] - Av @ Zv[:, avail], "fro") ** 2)

            S_sum = S_sum + S_v                        # 与集中式相同的累加顺序
            # 通信记账：上传 S_v(dim×N) + e_v(8B)，下发 H(dim×N)
            comm_up += bytes_mat + 8
            comm_down += bytes_mat + 8

        # --- 服务器：聚合 + 全局更新 ---
        # ④ H = solve((λI+γD), λ·Σ S_v)：与集中式 updateH 完全相同的线性求解
        H = solver.solve(lam * S_sum)
        if params.h_simplex:
            from vendor.osmagc.core.updates import eproj_simplex_batch
            H = eproj_simplex_batch(H, 1.0)
        # ⑤ α_v = e_v / Σ e_v：只汇总标量，不涉及矩阵
        total = e.sum()
        alpha = e / total if total > 0 else np.full(V, 1.0 / V)

        H_list.append(H.copy())
        comm_rounds = it

    comm_total = comm_up + comm_down
    labels = _honest_readout(H_list[-1], ds.n_classes, seed, n_init)

    detail = {
        "runtime_s": float(time.time() - t0),
        "norm": NORM,
        "partition": part.shape,
        "n_clients": int(V),
        "alpha": [float(x) for x in alpha],
        "num_pixel": [int(x) for x in params.num_pixel],
        "lam": float(lam),
        "gamma": float(gamma),
        "max_iter": int(params.max_iter),
        "dim": int(dim),
        "comm_bytes": int(comm_total),
        "comm_total_bytes": int(comm_total),
        "comm_up_bytes": int(comm_up),
        "comm_down_bytes": int(comm_down),
        "comm_rounds": int(comm_rounds),
        "comm_per_client_per_round": int(bytes_mat + 8),
        "comm_init_bytes": int(V * bytes_mat),
        "comm_bytes_with_init": int(comm_total + V * bytes_mat),
        "comm_accounting": "float64; 每轮每客户端: 上传 S_v(dim×N)+e_v(8B), "
                           "下发 H(dim×N); 初始化 H0 广播单列另计",
    }
    if return_H:
        detail["H_list"] = H_list
    return labels, detail
