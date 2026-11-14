"""有限差分算子，对应官方 MATLAB 的 `funs/construct_D.m`。

构造 D = Dx'Dx + Dy'Dy，其中 Dx / Dy 是水平 / 垂直方向的前向差分算子。
用于 OSAR（正交空间感知正则，论文 Eq.(4)(5)）：

    R(H) = ||Dx H'||_F^2 + ||Dy H'||_F^2 = Tr(H D H')

⚠️ **索引约定**：原 MATLAB 用的是 **列优先（Fortran order）** 的像素线性索引
（`construct_D.m:13,25` 的 `indices_left = (cols-1)*H + rows`）。
即：像素 (row, col) —— 0-based —— 的线性下标是 `col*H + row`。
这必须与 `preData` 里 `reshape(data, nRow*nCol, dim)` 的展开顺序一致，
否则 D 与 H/Z 的像素顺序错位，OSAR 会在错误方向上做平滑，结果直接崩。

原脚本把这个尺寸写死成 MUUFL（`osmagc.m:36` 的 `construct_D(325, 220)`），
这里做成参数，便于换数据集。
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def construct_D(H: int, W: int) -> sp.csr_matrix:
    """构造 D = Dx'Dx + Dy'Dy（N×N 稀疏，N = H*W，列优先像素序）。

    Parameters
    ----------
    H : int
        图像高（行数）。
    W : int
        图像宽（列数）。

    Returns
    -------
    scipy.sparse.csr_matrix
        N×N 对称稀疏矩阵。对任意 x（长度 N），
        `x @ D @ x` == 所有水平相邻对的差平方和 + 所有垂直相邻对的差平方和。
    """
    if H <= 0 or W <= 0:
        raise ValueError(f"H, W 必须为正整数，实际 ({H}, {W})")

    n = H * W

    # --- Dx: 水平差分 ---
    # 列优先下，向右移动一列 = 线性下标 +H
    # 共 H*(W-1) 条水平边：起点线性下标 k = 0 .. H*(W-1)-1，终点 k+H
    n_dx = H * (W - 1)
    if n_dx > 0:
        k = np.arange(n_dx)
        dx_rows = np.concatenate([k, k])
        dx_cols = np.concatenate([k, k + H])
        dx_vals = np.concatenate([-np.ones(n_dx), np.ones(n_dx)])
        Dx = sp.csr_matrix((dx_vals, (dx_rows, dx_cols)), shape=(n_dx, n))
    else:
        Dx = sp.csr_matrix((0, n))

    # --- Dy: 垂直差分 ---
    # 列优先下，向下移动一行 = 线性下标 +1
    # 但每列的最后一行没有向下的邻居，所以不是简单的 0..(H-1)*W-1
    # 对应 MATLAB 的 cols = repmat(1:W, H-1, 1); rows = repmat((1:H-1)', 1, W);
    # 按列优先展开：外层 c（列），内层 r（行 1..H-1）
    n_dy = (H - 1) * W
    if n_dy > 0:
        idx = np.arange(n_dy)
        c = idx // (H - 1)          # 列号 0..W-1
        r = idx % (H - 1)           # 行号 0..H-2
        m_top = c * H + r           # 上邻像素的列优先线性下标
        m_bot = m_top + 1           # 下邻
        dy_rows = np.concatenate([idx, idx])
        dy_cols = np.concatenate([m_top, m_bot])
        dy_vals = np.concatenate([-np.ones(n_dy), np.ones(n_dy)])
        Dy = sp.csr_matrix((dy_vals, (dy_rows, dy_cols)), shape=(n_dy, n))
    else:
        Dy = sp.csr_matrix((0, n))

    D = (Dx.T @ Dx + Dy.T @ Dy).tocsr()
    return D


def construct_D_literal(H: int, W: int) -> sp.csr_matrix:
    """`construct_D` 的字面翻译版，仅用于单元测试交叉验证。

    完全照抄 MATLAB 的 repmat / 冒号索引写法，可读性差但便于与原码逐句对照。
    """
    n = H * W

    # ---- Dx ----
    # indices_left = (cols-1)*H + rows, cols 是 H×(W-1)，rows 是 H×(W-1)
    cols = np.tile(np.arange(1, W), (H, 1))          # 1..W-1
    rows = np.tile(np.arange(1, H + 1).reshape(-1, 1), (1, W - 1))
    indices_left = (cols - 1) * H + rows             # 1-based
    indices_right = indices_left + H
    # MATLAB 的 (:)' 是列优先展开；这里用 order='F'
    jx = np.concatenate([indices_left.ravel(order="F"),
                         indices_right.ravel(order="F")]) - 1     # 转 0-based
    ix = np.tile(np.arange(H * (W - 1)), 2)
    vx = np.concatenate([-np.ones(H * (W - 1)), np.ones(H * (W - 1))])
    Dx = sp.csr_matrix((vx, (ix, jx)), shape=(H * (W - 1), n))

    # ---- Dy ----
    cols = np.tile(np.arange(1, W + 1), (H - 1, 1))
    rows = np.tile(np.arange(1, H).reshape(-1, 1), (1, W))
    indices_top = (cols - 1) * H + rows
    indices_bottom = indices_top + 1
    jy = np.concatenate([indices_top.ravel(order="F"),
                         indices_bottom.ravel(order="F")]) - 1
    iy = np.tile(np.arange((H - 1) * W), 2)
    vy = np.concatenate([-np.ones((H - 1) * W), np.ones((H - 1) * W)])
    Dy = sp.csr_matrix((vy, (iy, jy)), shape=((H - 1) * W, n))

    return (Dx.T @ Dx + Dy.T @ Dy).tocsr()
