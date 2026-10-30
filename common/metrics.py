"""统一聚类评价指标 —— ACC / NMI / ARI / Purity。

设计目标
--------
让 **所有算法**（Fed-MVKM、FedRSMVC、OSMAGC、EFIMVC、Local-only ...）的输出标签
都走同一个函数计算，否则表格里的数字互不可比。

口径来源：`D:\\cc\\OSMAGC\\osmagc\\metrics.py`（遥感侧参考实现，对应官方 MATLAB
`funs/evaluate_results_clustering.m`）。与 `Federated-MVC/cal_metric.py` 对拍一致。

四个坑（已在两边源码核对过）
----------------------------
1. **ACC 必须先做 `best_map`（匈牙利最优标签置换）**，否则聚类编号的任意排列会把
   ACC 打到随机水平。
2. **NMI 是算术平均归一化 `2·MI/(H_pred + H_true)`**（= sklearn 的
   `average_method='arithmetic'`，也是 sklearn 的默认值）。注意 OSMAGC 源码里变量
   名叫 `NMI_sqrt` 是误导，实现是算术平均。
3. **ARI 是 Hubert–Arabie**，等价于 sklearn 的 `adjusted_rand_score`。
4. **Purity 是「每个预测簇取占比最大的真值类，累加后除以 N」**，
   注意它**不是**置换后的 ACC（虽然形式接近）——Purity 允许两个簇映射到同一个真值类。

与现有实现的差异（本模块有意修正）
----------------------------------
- `Federated-MVC/cal_metric.py::purity_score` 会**原地改写调用方的 `y_true` 数组**
  （`y_true[y_true==labels[k]] = ordered_labels[k]`），同一数组二次调用结果即错。
  本模块全部为纯函数，绝不修改入参。
- 本模块用 bincount 构造列联表，O(N) 且无 Python 双层循环。

用法
----
>>> from common.metrics import evaluate
>>> evaluate(y_true, y_pred)
{'ACC': 0.612, 'NMI': 0.487, 'ARI': 0.400, 'Purity': 0.803}
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

METRIC_ORDER = ("ACC", "NMI", "ARI", "Purity")


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _as_int_labels(x) -> np.ndarray:
    """转成一维 int64，并做稠密化（去除标签空洞），不改动入参。"""
    x = np.asarray(x).ravel()
    if x.dtype.kind == "f":
        # 允许浮点标签，但要求是整数值（防止把 NaN 当标签）
        if not np.all(np.isfinite(x)):
            raise ValueError("标签中含 NaN/Inf，请先剔除背景像素")
        x = x.astype(np.int64)
    else:
        x = x.astype(np.int64)
    uniq = np.unique(x)
    if uniq.size == 0:
        raise ValueError("标签为空")
    # 稠密化：唯一值 -> 0..k-1（保持原有顺序）
    if uniq.size != int(uniq.max()) - int(uniq.min()) + 1:
        x = np.searchsorted(uniq, x).astype(np.int64)
    elif uniq.min() != 0:
        x = x - uniq.min()
    return x


def contingency(y_true, y_pred) -> np.ndarray:
    """列联表 `C[i, j] = #{pred == i 且 true == j}`。

    形状 `(n_pred_clusters, n_true_classes)`。与 OSMAGC 的 `CM(i,j)` 一致。
    """
    t = _as_int_labels(y_true)
    p = _as_int_labels(y_pred)
    if t.size != p.size:
        raise ValueError(f"长度不一致：true={t.size} vs pred={p.size}")
    n_t = int(t.max()) + 1
    n_p = int(p.max()) + 1
    flat = np.bincount(p * n_t + t, minlength=n_p * n_t)
    return flat.reshape(n_p, n_t).astype(np.int64)


def best_map(y_true, y_pred) -> np.ndarray:
    """匈牙利最优标签置换，返回置换后的 `y_pred`（不改动入参）。

    对应 OSMAGC 的 `best_map` / `evaluate_results_clustering.m:317`。
    """
    t = _as_int_labels(y_true)
    p = _as_int_labels(y_pred)
    C = contingency(t, p)                     # (n_pred, n_true)
    row, col = linear_sum_assignment(-C)       # row: 预测簇, col: 真值类
    mapping = np.arange(int(p.max()) + 1, dtype=np.int64)
    mapping[row] = col
    return mapping[p]


# --------------------------------------------------------------------------- #
# 四个指标
# --------------------------------------------------------------------------- #
def acc(y_true, y_pred) -> float:
    """Hungarian 置换后的聚类精度 (OA)。"""
    t = _as_int_labels(y_true)
    p = _as_int_labels(y_pred)
    C = contingency(t, p)
    row, col = linear_sum_assignment(-C)
    return float(C[row, col].sum() / C.sum())


def nmi(y_true, y_pred) -> float:
    """算术平均归一化互信息 `2·MI/(H_pred + H_true)`（与 sklearn 默认一致）。"""
    C = contingency(y_true, y_pred).astype(np.float64)
    N = C.sum()
    if N == 0:
        return 0.0
    a = C.sum(axis=1)          # 预测簇边缘
    b = C.sum(axis=0)          # 真值类边缘

    nz = C > 0
    # MI = sum n_ij/N * log(N * n_ij / (a_i * b_j))
    mi = float(
        np.sum(C[nz] / N * np.log(N * C[nz] / (a[:, None] * b[None, :])[nz]))
    )

    def _ent(m):
        m = m[m > 0] / N
        return float(-np.sum(m * np.log(m)))

    denom = _ent(a) + _ent(b)
    if denom == 0:
        # 退化：真值和预测都只有单一类别。此时两者熵均为 0，分母为 0。
        # 约定取 1.0（完全一致），与 sklearn 一致；MATLAB/OSMAGC 原码此处返回 0。
        # 该分支在实际评测中不会触发（聚类数恒等于类别数且 > 1）。
        return 1.0
    return 2.0 * mi / denom


def ari(y_true, y_pred) -> float:
    """调整兰德指数 (Hubert–Arabie)，等价 sklearn.metrics.adjusted_rand_score。"""
    C = contingency(y_true, y_pred).astype(np.float64)
    N = C.sum()

    def _c2(x):
        return x * (x - 1.0) / 2.0

    sum_ij = _c2(C).sum()
    sum_a = _c2(C.sum(axis=1)).sum()
    sum_b = _c2(C.sum(axis=0)).sum()
    total = _c2(N)
    if total == 0:
        return 0.0
    expected = sum_a * sum_b / total
    denom = (sum_a + sum_b) / 2.0 - expected
    if denom == 0:
        # 退化：两侧均为单一簇。约定取 1.0，与 sklearn 一致（同 NMI 的说明）。
        return 1.0
    return float((sum_ij - expected) / denom)


def purity(y_true, y_pred) -> float:
    """纯度 = (Σ_簇 max_类 |簇∩类|) / N。

    允许两个预测簇落到同一个真值类，因此与 ACC 不同。
    """
    C = contingency(y_true, y_pred)
    return float(C.max(axis=1).sum() / C.sum())


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred) -> dict:
    """一次返回四个指标。

    Returns
    -------
    dict
        `{'ACC': float, 'NMI': float, 'ARI': float, 'Purity': float}`，取值 [0,1]。
    """
    return {
        "ACC": acc(y_true, y_pred),
        "NMI": nmi(y_true, y_pred),
        "ARI": ari(y_true, y_pred),
        "Purity": purity(y_true, y_pred),
    }


def evaluate_at_mask(y_true_map, y_pred_map, mask) -> dict:
    """在 `mask` 指定的像素上评测（用于「全体像素参与聚类、只在有标注像素上打分」）。

    Parameters
    ----------
    y_true_map, y_pred_map : np.ndarray
        与影像同形状的标签图。
    mask : np.ndarray (bool)
        参与评测的像素（通常为 `gt > 0`）。
    """
    y_true_map = np.asarray(y_true_map)
    y_pred_map = np.asarray(y_pred_map)
    m = np.asarray(mask, dtype=bool)
    if y_true_map.shape != y_pred_map.shape:
        raise ValueError(
            f"真值/预测图形状不一致：{y_true_map.shape} vs {y_pred_map.shape}"
        )
    return evaluate(y_true_map[m], y_pred_map[m])
