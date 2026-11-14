"""OSMAGC 主优化循环，对应官方 MATLAB 的 `funs/opti.m`。

论文 Algorithm 1 的迭代顺序：
    A -> R -> Z -> H -> α      （共 max_iter 轮，每轮把 H 存进 pYlist）

官方 `osmagc.m` 在跑完 max_iter 轮后，对 pYlist 里的**每一个** H 都做一次
k-means 和评估，最后**取 ACC 最大的那一轮**作为报告值（见 NOTES.md D5）。
本模块只负责产生 pYlist，评估交给上层脚本。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..data.loader import Dataset, findindex
from .init import initA, initH, initZ
from .predata import predata
from .updates import HUpdater, updateA, updateBeta, updateH, updateRv, updateZv


@dataclass
class OSMAGCConfig:
    """单次运行的完整配置。

    未标注"论文"的项都对应 `osmagc.m` 里的 `temp_*` 变量。
    """

    num_pixel: list[int]              # temp_num_Pixel 的一行，如 [50, 100]
    lam: float = 1e-3                 # temp_lambda
    gamma: float = 1e-3               # temp_gamma
    dim: int | None = None            # temp_dim，None 表示取类别数 c
    max_iter: int = 10                # max_iter
    # --- 论文与代码不一致处的开关（见 NOTES.md） ---
    initz_k: str | int = "c"          # D2: 'c' 跟随代码 / 传整数则用该值
    h_simplex: bool = True            # D1: True 跟随代码（论文没写这一步）
    h_denom_v: bool = False           # D10: False 跟随代码（论文写的是带 V 的）
    pca_sign: int = 1                 # D7: 第一主成分符号（实测无影响，见 predata）
    mapminmax_axis: int = 1           # D13: 1=按行(MATLAB字面) / 0=按列(按波段)
    # --- 消融开关（论文 Table 4） ---
    sagi_mode: str = "ers"            # V1: 'kmeans' 表示去掉 SAGI
    sacfl_fixed: bool = False         # V3: True 表示固定对齐矩阵 R，即去掉 SACFL
    verbose: bool = True


@dataclass
class OSMAGCResult:
    """一次运行的产物。"""

    H_list: list[np.ndarray]          # max_iter 个 (dim, N)，对应 pYlist
    alpha: np.ndarray                 # 最终视图权重
    sp_labels: list[np.ndarray]       # V 个 (H, W) 超像素标签
    num_pixel: list[int]
    lam: float
    gamma: float
    dim: int
    config: OSMAGCConfig
    diagnostics: dict = field(default_factory=dict)


def build_inputs(ds: Dataset):
    """构造 `opti` 的输入：X（已 L2 归一化）、Q、可用像素视图 uX、unm。"""
    X, Q = findindex(ds.flat_data(), ds.index)
    V = ds.V
    q = [Q[:, v].astype(np.float64) for v in range(V)]        # (N,) 每个视图
    uX = [X[v][:, q[v] == 1] for v in range(V)]               # (Bv, unm_v)
    unm = np.array([int(q[v].sum()) for v in range(V)])
    return X, Q, q, uX, unm


def opti(
    X: list[np.ndarray],
    y: np.ndarray,
    num_pixel: list[int],
    dim: int,
    lam: float,
    gamma: float,
    Q: np.ndarray,
    index: list[np.ndarray],
    iter_num: int,
    sp_label: list[np.ndarray],
    D: sp.spmatrix,
    cfg: OSMAGCConfig,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    """对应 `funs/opti.m`，返回 (H 列表, 最终 alpha, 诊断信息)。"""
    V = len(X)
    N = len(y)
    c = len(np.unique(y)) - 1

    alpha = np.ones(V) / V
    q = [Q[:, v].astype(np.float64) for v in range(V)]
    uX = [X[v][:, q[v] == 1] for v in range(V)]
    unm = np.array([int(q[v].sum()) for v in range(V)])

    # D2：initZ 的近邻数 k
    k_initz = c if cfg.initz_k == "c" else int(cfg.initz_k)
    if cfg.initz_k != "c" and cfg.verbose:
        print(f"  [开关] initz_k 用独立值 {k_initz}（跟随论文，而非代码的 c={c}）")

    A = initA(X, sp_label, num_pixel)
    Z = initZ(uX, A, k_initz, index, unm, N)
    H = initH(N, dim)
    R = [None] * V

    solver = HUpdater(D, lam, gamma, V, use_v_factor=cfg.h_denom_v)

    H_list: list[np.ndarray] = []
    diag = {"alpha_hist": [], "residual_hist": []}

    for it in range(1, iter_num + 1):
        A = updateA(X, Z, q)
        # V3（w/o SACFL）：对齐矩阵只在第一轮算一次，之后固定不变
        if not (cfg.sacfl_fixed and it > 1):
            R = updateRv(V, Z, H)
        # ⚠️ 2026-10-07：updateZv 会新建 3 个 (Mv,N)（Salinas 下各 ~178 MB）。
        # 不先释放旧 Z 的话，新旧同时驻留，这一段就是 2×533 MB。旧 Z 在上面
        # 两步之后已无人引用（updateH/updateBeta 用的是下面新赋的 Z）。
        # **只影响分配时机，不改变任何一次计算**，数值逐位不变（已实测）。
        del Z
        Z = updateZv(A, X, N, R, H, lam, alpha, q)
        H = updateH(R, Z, lam, solver, h_simplex=cfg.h_simplex)
        alpha = updateBeta(X, A, Z, q)

        H_list.append(H.copy())
        diag["alpha_hist"].append(alpha.copy())

        if cfg.verbose:
            ev = [
                float(np.linalg.norm(X[v][:, q[v] == 1] - A[v] @ Z[v][:, q[v] == 1], "fro") ** 2)
                for v in range(V)
            ]
            diag["residual_hist"].append(ev)
            print(
                f"    iter {it:2d}/{iter_num}  H: [{H.min():+.3f}, {H.max():+.3f}] "
                f"sum/col={H.sum(axis=0).mean():.4f}  "
                f"alpha=[{', '.join(f'{a:.3f}' for a in alpha)}]  "
                f"e=[{', '.join(f'{x:.3f}' for x in ev)}]"
            )

    diag["alpha_hist"] = np.array(diag["alpha_hist"])
    return H_list, alpha, diag


def run_osmagc(ds: Dataset, cfg: OSMAGCConfig) -> OSMAGCResult:
    """完整跑一次 OSMAGC：preData -> opti。"""
    from .operators import construct_D

    if cfg.dim is None:
        dim = ds.c
    else:
        dim = int(cfg.dim)

    if cfg.verbose:
        print(
            f"  [配置] 数据集={ds.name} H={ds.H} W={ds.W} V={ds.V} N={ds.N} c={ds.c}\n"
            f"         num_pixel={cfg.num_pixel} lambda={cfg.lam:g} gamma={cfg.gamma:g} "
            f"dim={dim} max_iter={cfg.max_iter}\n"
            f"         initz_k={cfg.initz_k} h_simplex={cfg.h_simplex} "
            f"h_denom_v={cfg.h_denom_v} pca_sign={cfg.pca_sign:+d}"
        )

    pre = predata(
        ds.cubes, cfg.num_pixel,
        pca_sign=cfg.pca_sign, mapminmax_axis=cfg.mapminmax_axis,
        sagi_mode=cfg.sagi_mode,
    )
    if cfg.verbose:
        n_lab = [len(np.unique(l)) for l in pre.sp_labels]
        print(f"  [preData] ERS 超像素标签数 = {n_lab}（目标 {cfg.num_pixel}）")

    X, Q, q, uX, unm = build_inputs(ds)
    D = construct_D(ds.H, ds.W)

    if cfg.verbose:
        print(f"  [构造] D: shape={D.shape} nnz={D.nnz}；可用像素数={unm.tolist()}")

    H_list, alpha, diag = opti(
        X, ds.gt, cfg.num_pixel, dim, cfg.lam, cfg.gamma,
        Q, ds.index, cfg.max_iter, pre.sp_labels, D, cfg,
    )

    return OSMAGCResult(
        H_list=H_list,
        alpha=alpha,
        sp_labels=pre.sp_labels,
        num_pixel=list(cfg.num_pixel),
        lam=cfg.lam,
        gamma=cfg.gamma,
        dim=dim,
        config=cfg,
        diagnostics=diag,
    )
