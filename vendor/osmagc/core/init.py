"""变量初始化，对应官方 MATLAB 的 `funs/initA.m` / `funs/initZ.m` / `funs/initH.m`。

⚠️ 两个关键细节（都影响结果，见 NOTES.md D2）：

1. **`initZ` 的近邻数 k 传的是类别数 c**
   `funs/opti.m:17`：`Zv = initZ(uX, A, c, V, index, unm, N);`
   形参 k 对应论文 Eq.(1) 里的 "k nearest anchors"，实参给的是 `c`（类别数）。
   MUUFL c=11 → 每个像素取最近 12 个锚点。本模块用 `initz_k` 开关控制。

2. **超像素标签在逻辑索引时按列优先展平**
   `initA.m` 里 `X(:, label==Class(i))`，label 是 (H,W) 矩阵而 X 是 (B,N)。
   MATLAB 允许逻辑索引数组的形状与维度大小不一致（只要元素数相同），
   并按**列优先**顺序解释。所以这里统一 `reshape(-1, order='F')`。
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
from scipy.spatial.distance import cdist


def _econ_svd(M: np.ndarray):
    """MATLAB `svd(M, 'econ')` 的等价实现（走 LAPACK，符号约定与 MATLAB 一致）。"""
    return sla.svd(M, full_matrices=False)


def initA(X: list[np.ndarray], sp_label: list[np.ndarray], num_pixel: list[int]) -> list[np.ndarray]:
    """对应 `funs/initA.m`：用超像素内像素均值作为锚点。

    Parameters
    ----------
    X : list[np.ndarray]
        V 个 (Bv, N) 数组（`findindex` 输出的**已 L2 归一化**数据）。
    sp_label : list[np.ndarray]
        V 个 (H, W) 超像素标签（1-based）。
    num_pixel : list[int]
        V 个锚点数 Mv。

    Returns
    -------
    list[np.ndarray]
        V 个 (Bv, Mv) 锚点矩阵。
    """
    A = []
    for v in range(len(X)):
        Xv = np.asarray(X[v], dtype=np.float64)
        # 逻辑索引按列优先展平（对应 MATLAB 的行为）
        lab = np.asarray(sp_label[v]).reshape(-1, order="F")
        m = int(num_pixel[v])
        classes = np.unique(lab)

        means = np.zeros((Xv.shape[0], m), dtype=np.float64)
        for i, cls in enumerate(classes):
            if i >= m:
                break
            sub = lab == cls
            if sub.any():
                means[:, i] = Xv[:, sub].mean(axis=1)
        means[np.isnan(means)] = 0.0
        A.append(means)
    return A


def initZ(
    X: list[np.ndarray],
    anchor: list[np.ndarray],
    k: int,
    index: list[np.ndarray],
    unm: np.ndarray,
    N: int,
) -> list[np.ndarray]:
    """对应 `funs/initZ.m`：按 Eq.(1) 构造初始锚图。

    Parameters
    ----------
    X : list[np.ndarray]
        V 个 (Bv, **unm[v]**) 数组 —— 只含各视图**可用**像素。
    anchor : list[np.ndarray]
        V 个 (Bv, Mv) 锚点（`initA` 的输出）。
    k : int
        近邻数。官方传的是**类别数 c**（见 `opti.m:17`）。
    index : list[np.ndarray]
        V 个可用像素的 0-based 索引。
    unm : np.ndarray
        (V,) 各视图可用像素数。
    N : int
        总像素数。

    Returns
    -------
    list[np.ndarray]
        V 个 (Mv, N) 初始锚图；缺失像素所在列为 0。
    """
    V = len(X)
    # ⚠️ 2026-10-07：把原来"先攒齐 Z_dense 再散回"的两段循环合并成一段。
    # 原写法会让 3 个视图的 Z_dense（各 ~178 MB）和 3 个 Z_new（各 ~178 MB）
    # 同时驻留，这一段峰值 ~1.07 GB。合并后每个视图算完立刻散回并释放，
    # 峰值降到 ~710 MB。
    # **本改动只调整分配/释放时机，不改变任何一次 BLAS 调用或数组内容**，
    # 数值逐位不变（已实测 ACC/ARI 完全一致）。
    out = []
    for v in range(V):
        Xv = np.asarray(X[v], dtype=np.float64)         # (Bv, unm)
        Av = np.asarray(anchor[v], dtype=np.float64)    # (Bv, Mv)
        num, num_anchor = Xv.shape[1], Av.shape[1]

        if k + 1 > num_anchor:
            raise ValueError(
                f"视图 {v}: 近邻数 k+1 = {k + 1} 超过锚点数 {num_anchor}。"
                f"若 k 取类别数 c，需保证 Mv >= c+1。"
            )

        # 平方欧氏距离 (unm, Mv)
        dist = cdist(Xv.T, Av.T, metric="sqeuclidean")

        # 取最近的 k+1 个锚点
        idx_sorted = np.argsort(dist, axis=1, kind="stable")[:, : k + 1]
        d = np.take_along_axis(dist, idx_sorted, axis=1)   # (num, k+1) 升序

        # Eq.(1)：分母 k*d_{k+1} - sum(d_{1..k}) + eps
        denom = k * d[:, k] - d[:, :k].sum(axis=1) + np.finfo(np.float64).eps
        numer = d[:, k][:, None] - d
        zvals = numer / denom[:, None]

        Zv = np.zeros((num, num_anchor), dtype=np.float64)
        np.put_along_axis(Zv, idx_sorted, zvals, axis=1)

        # 立刻散回 (N, Mv) 再转置成 (Mv, N)，不保留全量 Z_dense
        cur_unm = int(unm[v])
        Z_new = np.zeros((N, num_anchor), dtype=np.float64)
        if cur_unm > 0:
            Z_new[np.asarray(index[v], dtype=np.int64), :] = Zv[:cur_unm, :]
        del Zv, dist, idx_sorted, d, zvals
        out.append(Z_new.T)          # (Mv, N)
    return out


def initH(N: int, dim: int) -> np.ndarray:
    """对应 `funs/initH.m`：对 [I_dim | 0] 做 econ SVD，返回 (dim, N) 正交行矩阵。"""
    p = np.zeros((dim, N), dtype=np.float64)
    p[:, :dim] = np.eye(dim)
    U, _, _ = _econ_svd(p.T)      # p.T 是 (N, dim)
    return U.T                     # (dim, N)
