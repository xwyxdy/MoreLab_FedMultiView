"""不完整数据的缺失模式生成，对应官方 MATLAB 的 `data/gen_misssing_muufl.m`。

⚠️ 官方的 `gen_misssing_muufl.m:28` 有语法错误（`x2 = t,ranspose(...)` 多了个逗号），
脚本无法直接运行。但缺失规则本身是清楚的，本模块按该规则精确复刻。

规则（见 `gen_misssing_muufl.m:24-31`）
--------------------------------------
设 N = H*W 为总像素数，V 为视图数，rate 为缺失率：

    missr = fix(rate * N / V)          # MATLAB 的 fix = 向零取整

    view 1 缺 [1,           missr      ]     (1-based)
    view 2 缺 [missr+1,     2*missr    ]
    ...
    view v 缺 [(v-1)*missr+1, v*missr  ]

即：按**列优先顺序连续分配、各视图互不重叠**。

"缺失率" 的含义
----------------
rate 是**相对总像素数 N** 的：总缺失条目数 = V * missr = rate * N。

即 rate=0.5 表示 "N 个像素中，平均有 0.5*N 个 (视图,像素) 观测是缺失的"。
换成单个视图的口径是 rate/V，换成全体条目的口径是 rate/V。

已验证（`scripts/verify_missing.py`）：MUUFL (V=2) 下 missr = fix(0.5*71500/2) = 17875，
每个视图各缺 17875 个（占该视图 25%），合计 35750 = 0.5 * 71500 ✓

⚠️ 缺失不是随机散布的，而是**空间整块**的：MUUFL@0.5 下 view1 缺最左 55 列、
view2 缺第 56-110 列。这与论文 Fig.1 描述的"传感器故障 / 云遮挡"场景吻合。
"""

from __future__ import annotations

import math

import numpy as np


def compute_missr(n: int, V: int, rate: float) -> int:
    """每视图缺失的像素数：`fix(rate * n / V)`（MATLAB fix 向零取整）。"""
    if V <= 0:
        raise ValueError(f"V 必须为正整数，实际 {V}")
    return int(math.trunc(rate * n / V))


def make_missing_indices(n: int, V: int, rate: float) -> list[np.ndarray]:
    """生成各视图的**可用**像素索引（0-based）。

    Parameters
    ----------
    n : int
        总像素数 N = H*W。
    V : int
        视图数。
    rate : float
        缺失率（全体条目口径）。

    Returns
    -------
    list[np.ndarray]
        长度 V，第 v 个元素是该视图可用像素的 0-based 索引（升序）。

    Examples
    --------
    >>> idx = make_missing_indices(71500, 2, 0.5)
    >>> idx[0][0], idx[0][-1]     # view1 可用区间
    (17875, 71499)
    >>> idx[1][0], idx[1][17874], idx[1][17875]
    (0, 17874, 35750)
    """
    missr = compute_missr(n, V, rate)
    if V * missr > n:
        raise ValueError(
            f"缺失参数非法：V*missr = {V * missr} > n = {n}（rate={rate}, V={V}）"
        )

    all_idx = np.arange(n)
    out: list[np.ndarray] = []
    for v in range(V):
        start = v * missr          # 0-based 起点
        stop = (v + 1) * missr
        # 对应 MATLAB: index = [(1:start) (stop+1:n)]'  的 0-based 形式
        out.append(np.concatenate([all_idx[:start], all_idx[stop:]]))
    return out


def make_missing_mask(n: int, V: int, rate: float) -> np.ndarray:
    """返回 (N, V) 的 uint8 可用掩码：1 = 该视图该像素可用（对应 MissingStatus）。"""
    mask = np.zeros((n, V), dtype=np.uint8)
    for v, idx in enumerate(make_missing_indices(n, V, rate)):
        mask[idx, v] = 1
    return mask
