"""交替优化中的各变量更新，对应官方 MATLAB 的 `funs/update*.m`。

论文与代码的差异见 NOTES.md，其中影响本模块的是：

- **D1**：`updateH` 在算出 Eq.(11) 之后，**还对 H 的每一列做了单纯形投影**
  （`updateH.m:11-13`）。论文正文完全没写这一步。开关 `h_simplex` 控制。
- **D10**：`updateH` 的分母是 `gamma*D + lambda*I`，**少了 V 因子**
  （`updateH.m:8`），而论文 Eq.(11) 写的是 `λVI + γD`。开关 `h_denom_v` 控制。
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .simplex import eproj_simplex_new

# 列分块大小：控制 (Mv, N) 级大数组的峰值内存。
# Berlin（N=820148）下单个 (100,N) float64 数组就是 656 MB，
# 不分块会同时开出好几个，内存直接上 8 GB。
#
# ⚠️ 2026-10-07 由 200_000 下调到 25_000。原因：200_000 这个阈值是为
# Berlin(N=820,148) 量身定的，**对 N < 200_000 的数据集（Salinas N=111,104、
# MUUFL、Trento 全部）分块逻辑一次都不会触发**——`F.shape[1] <= chunk` 直接
# 走 `_eproj_simplex_batch_core` 全尺寸路径，单视图瞬时开出 ~1.4 GB 临时量，
# 单次运行峰值 3.7 GB，把并行度锁死在 1。
# 分块只切输出列、每列独立求解，**数值不变**（已实测 ACC 逐位一致）。
COL_CHUNK = 25_000


def _econ_svd(M: np.ndarray):
    """MATLAB `svd(M, 'econ')`。"""
    return sla.svd(M, full_matrices=False)


# ------------------------------------------------------------------ 单纯形批量投影
def eproj_simplex_batch(F: np.ndarray, k: float = 1.0, chunk: int = 200_000) -> np.ndarray:
    """对 F 的**每一列**做单纯形投影（向量化版 `EProjSimplex_new`）。

    与逐个调用 `eproj_simplex_new` 数值等价。向量化用同一套 Newton 迭代，
    只对尚未收敛的列继续迭代；若某列迭代满 100 次仍未收敛，
    回退到标量版以保证与 MATLAB 完全一致的行为。

    ⚠️ **按列分块**：Berlin（N=820148）时 (Mv,N) 的 float64 数组单个就有 656 MB，
    迭代中会同时存在好几个（v0/v1/pos/临时量），合起来能到 8 GB。
    分块后峰值内存降到 `chunk × d` 量级。

    Parameters
    ----------
    F : np.ndarray
        (d, n)，每一列独立投影。
    k : float
        目标和。
    chunk : int
        每次处理的列数上限。

    Returns
    -------
    np.ndarray
        与 F 同形状。
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim == 1:
        return eproj_simplex_new(F, k)

    if F.shape[1] <= chunk:
        return _eproj_simplex_batch_core(F, k)

    out = np.empty_like(F)
    for s in range(0, F.shape[1], chunk):
        e = min(s + chunk, F.shape[1])
        out[:, s:e] = _eproj_simplex_batch_core(F[:, s:e], k)
    return out


def _eproj_simplex_batch_core(F: np.ndarray, k: float = 1.0) -> np.ndarray:
    """`eproj_simplex_batch` 的核心实现（不分块，列数应已受控）。"""
    d, n = F.shape
    v0 = F - F.mean(axis=0, keepdims=True) + k / d
    out = v0.copy()

    active = v0.min(axis=0) < 0.0
    if not active.any():
        return out

    lam = np.zeros(n, dtype=np.float64)
    it = 0
    while active.any() and it <= 100:
        cols = np.nonzero(active)[0]
        v1 = v0[:, cols] - lam[cols]
        pos = v1 > 0
        npos = pos.sum(axis=0)
        g = -npos.astype(np.float64)
        f = np.where(pos, v1, 0.0).sum(axis=0) - k

        with np.errstate(divide="ignore", invalid="ignore"):
            lam[cols] = lam[cols] - f / g

        # 收敛的列移出（MATLAB: while abs(f) > 1e-10）
        active[cols] = np.abs(f) > 1e-10
        it += 1

    out = np.maximum(v0 - lam, 0.0)

    # 未收敛的列回退到标量版，保证行为与 MATLAB 一致
    stuck = np.nonzero(active)[0]
    for j in stuck:
        out[:, j] = eproj_simplex_new(v0[:, j], k)

    return out


# ------------------------------------------------------------------------ updateA
def updateA(X: list[np.ndarray], Z: list[np.ndarray], q: list[np.ndarray]) -> list[np.ndarray]:
    """对应 `funs/updateA.m`：解 Eq.(8)，A = U V'（econ SVD）。

    注意：econ SVD 只保证 `A A' = I`（行正交），**不保证** `A'A = I`。
    LiDAR 视图只有 2 个波段，锚点却有 50 个时，论文 Eq.(2) 的 `A'A = I` 无法满足。
    原码照跑不误 —— 这是论文表述与实现的一处不一致。
    """
    A = []
    for v in range(len(X)):
        # 对应 MATLAB: M = X{v} .* q{v} * Z{v}'  （先按列乘掩码，再右乘 Z'）
        M = (X[v] * q[v]) @ Z[v].T
        U, _, Vt = _econ_svd(M)
        A.append(U @ Vt)
    return A


# ----------------------------------------------------------------------- updateRv
def updateRv(V: int, Z: list[np.ndarray], H: np.ndarray) -> list[np.ndarray]:
    """对应 `funs/updateRv.m`：解 Eq.(9)，R = U V'（rank-D 截断 SVD）。

    Rv 形状是 (dim, Mv)，满足 `Rv Rv' = I_dim`（论文 Eq.(3) 写成 `Rv'Rv = I`，
    在 dim < Mv 时维度上不可能成立，属笔误，见 NOTES.md D3）。
    """
    Rv = []
    for v in range(V):
        HZ = H @ Z[v].T                       # (dim, Mv)
        U, _, Vt = _econ_svd(HZ)
        Rv.append(U @ Vt)                     # (dim, Mv)
    return Rv


# ----------------------------------------------------------------------- updateZv
def updateZv(
    A: list[np.ndarray],
    X: list[np.ndarray],
    N: int,
    R: list[np.ndarray],
    H: np.ndarray,
    lam: float,
    alpha: np.ndarray,
    q: list[np.ndarray],
) -> list[np.ndarray]:
    """对应 `funs/updateZv.m`：解 Eq.(7)，逐列投影到单纯形。

        f_ij = [ (1/alpha_v) (A_v' X_v) ⊙ q_v + λ (R_v' H) ]_ij
               / [ (1/alpha_v) q_v + λ ]_ij
    """
    V = len(X)
    Z = []
    for v in range(V):
        Mv = A[v].shape[1]
        inv_a = 1.0 / alpha[v]
        qv = q[v]
        Zv = np.empty((Mv, N), dtype=np.float64)
        # 按列分块：避免同时开出多个 (Mv,N) 的大数组（Berlin 时单个就有 656 MB）
        for s in range(0, N, COL_CHUNK):
            e = min(s + COL_CHUNK, N)
            ax = A[v].T @ X[v][:, s:e]                      # (Mv, blk)
            rh = R[v].T @ H[:, s:e]                         # (Mv, blk)
            t2 = inv_a * qv[s:e] + lam                       # (blk,)
            F = (inv_a * (ax * qv[None, s:e]) + lam * rh) / t2[None, :]
            Zv[:, s:e] = eproj_simplex_batch(F, 1.0, chunk=COL_CHUNK)
        Z.append(Zv)
    return Z


# ------------------------------------------------------------------------ updateH
class HUpdater:
    """`updateH` 的线性求解器，带分解缓存。

    `funs/updateH.m:10` 的 `F = temp1 / temp2`（MATLAB 的 mrdivide）等价于
    `temp1 * inv(temp2)`，即解 `temp2 @ F.T = temp1.T`。
    由于 λ、γ、V、D 在整个迭代过程中不变，`temp2` 是**常量矩阵**，
    只需 LU 分解一次，后续每轮复用 —— 这也是 Berlin(N=820148) 能跑动的关键。
    """

    def __init__(
        self,
        D: sp.spmatrix,
        lam: float,
        gamma: float,
        V: int,
        use_v_factor: bool = False,
    ):
        n = D.shape[0]
        # h_denom_v=False 跟随代码(updateH.m:8)；True 跟随论文 Eq.(11)
        lam_coef = lam * (V if use_v_factor else 1)
        self.matrix = (gamma * D + lam_coef * sp.eye(n, format="csc")).tocsc()
        self.lu = splu(self.matrix)
        self.n = n

    def solve(self, temp1: np.ndarray) -> np.ndarray:
        """求解 temp2 @ X = temp1（按列），返回与 temp1 同形状 (dim, N)。"""
        rhs = np.asfortranarray(temp1.T)          # (N, dim)
        sol = self.lu.solve(rhs)                  # (N, dim)
        return np.ascontiguousarray(sol.T)        # (dim, N)


def updateH(
    R: list[np.ndarray],
    Z: list[np.ndarray],
    lam: float,
    solver: HUpdater,
    h_simplex: bool = True,
) -> np.ndarray:
    """对应 `funs/updateH.m`：解 Eq.(11)（+ 可选的单纯形投影）。

    Parameters
    ----------
    h_simplex : bool
        True（默认，跟随代码 `updateH.m:11-13`）对 H 的每列做单纯形投影；
        False 则只算 Eq.(11)（跟随论文正文）。
    """
    # 对应 MATLAB 的 `part1 = 0` 后累加（那边初值是标量，会自动定形为 (dim, N)）
    part1 = R[0] @ Z[0]
    for v in range(1, len(R)):
        part1 = part1 + R[v] @ Z[v]

    temp1 = lam * part1
    F = solver.solve(temp1)

    if h_simplex:
        return eproj_simplex_batch(F, 1.0)
    return F


# --------------------------------------------------------------------- updateBeta
def updateBeta(X: list[np.ndarray], A: list[np.ndarray], Z: list[np.ndarray], q: list[np.ndarray]) -> np.ndarray:
    """对应 `funs/updateBeta.m`：解 Eq.(13)，α_v = e_v / Σ e_v。

    其中 `e_v = ||X_v Q_v - A_v Z_v Q_v||_F^2` 是第 v 个视图的重建残差平方。
    """
    V = len(X)
    e = np.zeros(V, dtype=np.float64)
    for v in range(V):
        avail = q[v] == 1
        XQ = X[v][:, avail]
        ZQ = Z[v][:, avail]
        residual = XQ - A[v] @ ZQ
        e[v] = float(np.linalg.norm(residual, "fro") ** 2)

    total = e.sum()
    if total <= 0:
        return np.full(V, 1.0 / V)
    return e / total
