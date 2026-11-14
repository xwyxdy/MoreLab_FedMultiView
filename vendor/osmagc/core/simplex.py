"""单纯形投影，对应官方 MATLAB 的 `funs/EProjSimplex_new.m`。

求解：
    min  1/2 ||x - v||^2
    s.t. x >= 0,  1'x = k

被用在两处（都不在论文正文里）：
  - `funs/updateZv.m:9`  对 Z 的每一列投影（论文 Eq.(7) 有提，是 projection capped simplex）
  - `funs/updateH.m:13`  对 H 的每一列投影（⚠️ 论文 Eq.(11) 完全没写这一步，见 NOTES.md D1）

本实现逐行复刻原 MATLAB，包括其**迭代终止条件和除零行为**，以保证数值一致。
"""

from __future__ import annotations

import numpy as np


def eproj_simplex_new(v: np.ndarray, k: float = 1.0) -> np.ndarray:
    """把向量 v 投影到单纯形 {x : x >= 0, sum(x) = k} 上。

    Parameters
    ----------
    v : np.ndarray
        待投影向量（1D）。
    k : float
        目标和，默认 1（对应 MATLAB 的 `k = 1` 默认参数）。

    Returns
    -------
    np.ndarray
        与 v 同形状的投影结果。

    Notes
    -----
    原 MATLAB 用 while 迭代求 Lagrange 乘子。当 `min(v0) < 0` 时进入迭代，
    最多 100 次；若 100 次未收敛则直接取 `max(v1, 0)`。
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1:
        v = v.ravel()
    n = v.size
    if n == 0:
        return v.copy()

    # MATLAB: v0 = v - mean(v) + k/n
    v0 = v - v.mean() + k / n

    # MATLAB: if vmin < 0 ... else x = v0
    if v0.min() >= 0:
        return v0

    ft = 1
    lambda_m = 0.0
    f = 1.0
    v1 = v0

    # MATLAB: while abs(f) > 10^-10
    # 注意 g = -npos；npos 为 0 时 g = 0，MATLAB 会得到 Inf/NaN 并一直迭代到 ft>100。
    # 这里用 errstate 静默，保持与 MATLAB 相同的数值行为。
    with np.errstate(divide="ignore", invalid="ignore"):
        while abs(f) > 1e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = int(posidx.sum())
            g = -npos
            f = float(v1[posidx].sum()) - k

            if g == 0:
                # 与 MATLAB 的 0/0 = NaN 行为一致：lambda_m 变 NaN，
                # 下一轮 abs(f) 比较为 False 而退出（或迭代到上限）。
                lambda_m = np.nan
            else:
                lambda_m = lambda_m - f / g

            ft += 1
            if ft > 100:
                break

    # MATLAB: x = max(v1, 0)
    return np.maximum(v1, 0.0)
