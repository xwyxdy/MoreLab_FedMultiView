"""ETAP 的 Python 复现。

论文：Efficient Tensorized Multi-View Anchor Graph Clustering with Affinity
      Propagation for Remote Sensing Data（AAAI 2026，Zhang Yongshan 组）。

官方 MATLAB：github.com/ZhangYongshan/ETAP。本模块逐函数转写其 `main.m` + `funs/*.m`。

算法总览（与 OSMAGC 同源，但多两层）：
  1. 超像素（ERS）去噪 + 锚点初始化  —— 逐视图
  2. 锚图 Z（像素×锚点）             —— 逐视图
  3. 投影 W（把高维 HS 投影到低维子空间）—— 逐视图
  4. 压缩锚图 S（锚点×锚点）          —— 逐视图，但受全局 F 约束
  5. 张量 Schatten-p 范数（把各视图 S 堆成张量做 t-SVD）—— 全局（跨视图）
  6. 加权亲和传播出像素标签            —— 全局（加权和）

关键约定：像素展平用 Fortran（列优先），与平台 `ds.flatten_view("F")` 一致。
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp
from scipy.spatial.distance import cdist

# 复用 OSMAGC 的 ERS 与预处理（ERS 参数完全一致：lambda=0.5, sigma=5.0, kernel=0）
from ..osmagc.core.ers import mex_ers
from ..osmagc.core.predata import mapminmax, pca_coeff, mat2gray, im2uint8

EPS = np.finfo(np.float64).eps


# --------------------------------------------------------------------------- #
# 距离 / 投影到单纯形 / 特征分解
# --------------------------------------------------------------------------- #
def sqdist(a, b):
    """平方欧氏距离矩阵。a:(n,d), b:(m,d) -> (n,m)。对应 pdist2(...)^2。"""
    return cdist(a, b, metric="sqeuclidean")


def eproj_simplex(v, k=1.0):
    """min 0.5||x-v||^2 s.t. x>=0, 1'x=k。对应 EProjSimplex_new。"""
    n = v.shape[0]
    v0 = v - v.mean() + k / n
    vmin = v0.min()
    if vmin < 0:
        lam = 0.0
        f = 1.0
        ft = 0
        while abs(f) > 1e-10:
            v1 = v0 - lam
            pos = v1 > 0
            npos = pos.sum()
            g = -npos
            f = v1[pos].sum() - k
            lam = lam - f / g
            ft += 1
            if ft > 100:
                break
        x = np.maximum(v1, 0.0)
    else:
        x = v0
    return x


def eig1(A, c, isMax=1, isSym=1):
    """对应 funs/tools/eig1.m：返回 (前c个特征向量, 前c个特征值, 全部特征值)。"""
    A = np.asarray(A, dtype=np.float64)
    c = int(min(c, A.shape[0]))
    if isSym == 1:
        A = np.maximum(A, A.T)
        w, v = np.linalg.eigh(A)          # 对称：升序实特征值
    else:
        w, v = np.linalg.eig(A)           # 一般矩阵：可能复数
        w = np.real(w)                     # 广义特征值应为实，取实部排序
    if isMax == 0:
        order = np.argsort(w)
    else:
        order = np.argsort(w)[::-1]
    eigval = w[order[:c]]
    eigvec = v[:, order[:c]]
    eigval_full = w[order]
    return eigvec, eigval, eigval_full


# --------------------------------------------------------------------------- #
# 张量 Schatten-p 范数的近端算子（t-SVD + 广义软阈值）
# --------------------------------------------------------------------------- #
def solve_lp_w(y, lam, p):
    """广义软阈值（Schatten-p）。对应 solve_Lp_w.m。"""
    lam = float(lam)
    y = np.asarray(y, dtype=np.float64)
    x = np.zeros_like(y)
    tau = (2 * lam * (1 - p)) ** (1 / (2 - p)) + p * lam * (2 * (1 - p) * lam) ** ((p - 1) / (2 - p))
    i0 = np.abs(y) > tau
    if i0.any():
        y0 = y[i0]
        t = np.abs(y0)
        for _ in range(4):
            t = np.abs(y0) - p * lam * (t ** (p - 1))
        x[i0] = np.sign(y0) * t
    return x


def wshrink_obj_weight_lp(x, rho, sX, isWeight, mode, p):
    """对应 wshrinkObj_weight_lp.m。mode=3（top slice）时张量的管方向是第 1 维。"""
    X = x.reshape(sX, order="F")
    n1, n2, n3 = sX
    if mode == 3:
        Y = np.moveaxis(X, 0, -1)          # shiftdim(X,1): (n1,n2,n3)->(n2,n3,n1)
        N3 = n1
    else:
        Y = X
        N3 = n3

    Yhat = np.fft.fft(Y, axis=-1)

    def _shrink_s(s):
        if isWeight:
            C = np.sqrt(n3 * n2)
            weight = C / (s + EPS)
            return np.maximum(s - rho * weight, 0.0)
        return solve_lp_w(s, rho, p)

    endValue = N3 // 2 + 1
    for i in range(1, endValue + 1):       # i 为 1-based，对应 MATLAB 循环
        M = Yhat[..., i - 1]
        u, s, vt = np.linalg.svd(M, full_matrices=False)
        shat = _shrink_s(s)
        Yhat[..., i - 1] = (u * shat) @ vt
        if i > 1:
            j = N3 - i + 1                 # MATLAB 1-based n3-i+2 -> 0-based n3-i+1
            Yhat[..., j] = (np.conjugate(u) * shat) @ np.conjugate(vt)

    Y = np.fft.ifft(Yhat, axis=-1).real
    if mode == 3:
        X = np.moveaxis(Y, -1, 0)          # shiftdim(Y,2)
    else:
        X = Y
    return X.reshape(-1, order="F")


# --------------------------------------------------------------------------- #
# 超像素去噪（LED）
# --------------------------------------------------------------------------- #
def gen_anchor_adj(X, anchor, k, issymmetric=1):
    """对应 Gen_Achor_Adj.m：k 近邻软权重（X/anchor 每列一个样本）。"""
    n, m = X.shape[1], anchor.shape[1]
    D = sqdist(X.T, anchor.T)              # (n, m)
    k_used = min(k, m - 1)
    S = sp.csr_matrix((n, m))
    if k_used <= 0:
        return S
    idx = np.argsort(D, axis=1)[:, 1:k_used + 1]   # 跳过自身（最近的是自己）
    di = np.take_along_axis(D, idx, axis=1)        # (n, k_used)
    denom = k_used * di[:, -1:] - di.sum(axis=1, keepdims=True) + EPS
    vals = (di[:, -1:] - di) / denom
    rows = np.repeat(np.arange(n), k_used)
    S = sp.csr_matrix((vals.ravel(), (rows, idx.ravel())), shape=(n, m))
    if issymmetric:
        S = (S + S.T) / 2.0
    return S


def seg_im_class(Y, Ln):
    """对应 seg_im_class.m：按超像素标签分组像素。"""
    M, N, B = Y.shape
    Y_reshape = Y.reshape(M * N, B, order="F")
    Gt = Ln.reshape(1, -1, order="F")
    classes = np.unique(Gt)
    idx_list, y_list = [], []
    for c in classes:
        mask = (Gt == c)[0]
        idx_list.append(np.flatnonzero(mask))
        y_list.append(Y_reshape[mask])
    return idx_list, y_list


def led(data, lam, labels, dk):
    """局部嵌入去噪：在每个超像素块内做图平滑。对应 LED.m。"""
    nRow, nCol, dim = data.shape
    idx_list, y_list = seg_im_class(data, labels)
    A = np.zeros((nRow * nCol, dim))
    for idx, y in zip(idx_list, y_list):
        if y.shape[0] <= 1:
            A[idx] = y
            continue
        W = gen_anchor_adj(y.T, y.T, dk, 1)          # (L, L) 稀疏
        d = np.asarray(W.sum(axis=1)).ravel()
        L = sp.diags(d) - W
        M = lam * L + sp.eye(y.shape[0], format="csr")
        sol = sp.linalg.spsolve(M.tocsc(), y)
        A[idx] = np.asarray(sol).reshape(y.shape)    # 单波段时 spsolve 返回 1D，需 reshape
    return A.reshape(nRow, nCol, dim, order="F")


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #
def preprocess(cubes, t, dk, num_pixel):
    """对应 preDataqvzao.m：mapminmax -> PCA -> ERS 超像素 -> LED 去噪。

    Returns
    -------
    X : list[(B_v, N)]   去噪后数据，每列一个像素
    labels : list[(H, W)] 超像素标签（1-based）
    """
    X_out, labels_out = [], []
    for cube in cubes:
        cube = np.asarray(cube, dtype=np.float64)
        nRow, nCol, dim = cube.shape
        X = cube.reshape(nRow * nCol, dim, order="F")
        X = mapminmax(X, axis=1)                     # 逐像素归一化到 [-1,1]
        coeff = pca_coeff(X)
        y_pca = X @ coeff[:, 0]
        img = im2uint8(mat2gray(y_pca.reshape(nRow, nCol, order="F")))
        labels = mex_ers(img, int(num_pixel)) + 1
        newData = led(cube, t, labels, dk)
        X_denoised = newData.reshape(nRow * nCol, dim, order="F").T   # (dim, N)
        X_out.append(np.ascontiguousarray(X_denoised))
        labels_out.append(labels.astype(np.int32))
    return X_out, labels_out


# --------------------------------------------------------------------------- #
# 初始化
# --------------------------------------------------------------------------- #
def init_a(X, sp_labels, m):
    """对应 initA.m：超像素均值作锚点。返回 list[(B_v, m)]。"""
    A = []
    for v in range(len(X)):
        Xv = np.asarray(X[v], dtype=np.float64)
        lab = np.asarray(sp_labels[v]).reshape(-1, order="F")
        classes = np.unique(lab)
        means = np.zeros((Xv.shape[0], m))
        for i in range(min(len(classes), m)):
            sub = (lab == classes[i])
            means[:, i] = Xv[:, sub].mean(axis=1)
        means[np.isnan(means)] = 0.0
        A.append(means)
    return A


def _soft_knn_anchor(X, anchor, k, alpha_v):
    """构造锚图：投影/原始空间的 k+1 近邻软权重。返回 (N, m)。"""
    dist = sqdist(X.T, anchor.T) * (1.0 / alpha_v)     # (N, m)
    num, m = dist.shape
    k_used = min(k, m - 1)
    Z = np.zeros((num, m))
    idx = np.argsort(dist, axis=1)[:, :k_used + 1]
    di = np.take_along_axis(dist, idx, axis=1)          # (N, k+1) 升序
    numer = di[:, -1:] - di                             # d_{k+1} - d_j
    # 分母 = k*d_{k+1} - sum(d_1..d_k)（不含 d_{k+1}）
    denom = k_used * di[:, -1:] - di[:, :-1].sum(axis=1, keepdims=True) + EPS
    vals = numer / denom
    for j in range(k_used + 1):
        Z[np.arange(num), idx[:, j]] = vals[:, j]
    return Z


def init_z(X, anchor, k, alpha):
    """对应 initZ.m：无投影的初始锚图。返回 list[(N, m)]。"""
    V = len(X)
    Z = []
    for v in range(V):
        Z.append(_soft_knn_anchor(X[v], anchor[v], k, alpha[v]))
    return Z


def _init_w_detail(X, A, Z, r):
    """对应 initW_detail.m。"""
    St = X @ X.T
    H2 = X @ Z @ A.T
    Q = St - (H2 + H2.T) + A @ np.diag(Z.sum(axis=0)) @ A.T
    Q = (Q + Q.T) / 2.0
    H = np.linalg.lstsq(St, Q, rcond=None)[0]
    r = int(min(r, St.shape[0]))
    W, _, _ = eig1(H, r, 0, 0)
    W = np.real(W)
    W = W @ np.diag(1.0 / np.sqrt(np.diag(W.T @ W) + EPS))
    return W


def init_w(X, A, Z, r):
    """对应 initW.m。r 为逐视图投影维度列表。"""
    return [_init_w_detail(X[v], A[v], Z[v], r[v]) for v in range(len(X))]


# --------------------------------------------------------------------------- #
# 更新
# --------------------------------------------------------------------------- #
def update_z(X, anchor, k, W, alpha):
    """对应 updateZ.m：投影空间的锚图。返回 list[(N, m)]。"""
    V = len(X)
    Z = []
    for v in range(V):
        px = W[v].T @ X[v]                              # (r, N)
        pa = W[v].T @ anchor[v]                         # (r, m)
        Z.append(_soft_knn_anchor(px, pa, k, alpha[v]))
    return Z


def update_a(X, Z, S, alpha, beta):
    """对应 updateA.m。返回 list[(B_v, m)]。"""
    A = []
    for v in range(len(X)):
        alpha_v, beta_v = alpha[v], beta[v]
        Xv, Zv, Sv = X[v], Z[v], S[v]
        m = Zv.shape[1]
        Dz = np.diag(Zv.sum(axis=0))
        inv_term = beta_v * Dz + alpha_v * np.eye(m) - alpha_v * Sv - alpha_v * Sv.T + alpha_v * (Sv @ Sv.T)
        Av = (beta_v * Xv @ Zv) @ np.linalg.inv(inv_term)
        A.append(Av)
    return A


def update_alpha(W, X, A, Z):
    """对应 updateAlpha.m：视图权重。"""
    V = len(X)
    hv = np.zeros(V)
    for v in range(V):
        H2 = X[v] @ Z[v] @ A[v].T
        Q = X[v] @ X[v].T - (H2 + H2.T) + A[v] @ np.diag(Z[v].sum(axis=0)) @ A[v].T
        hv[v] = np.trace(W[v].T @ Q @ W[v])
    sq = np.sqrt(np.maximum(hv, 0.0))
    return sq / (sq.sum() + EPS)


def update_beta(A, S, W):
    """对应 updateBeta.m：压缩锚图权重。"""
    V = len(A)
    hv = np.zeros(V)
    for v in range(V):
        hv[v] = np.linalg.norm(W[v].T @ A[v] - W[v].T @ A[v] @ S[v], "fro")
    hv = np.maximum(hv, 1e-12)          # 防止某视图残差为 0 导致 beta=0
    return hv / (hv.sum() + EPS)


def update_f(S, beta, c):
    """对应 updateF.m：从加权拉普拉斯和取最小的 c 个特征向量。"""
    V = len(S)
    Ls_sum = None
    for v in range(V):
        Sv = (S[v] + S[v].T) / 2.0
        Ds = np.diag(Sv.sum(axis=0))
        Ls = Ds - Sv
        Ls_sum = (1.0 / beta[v]) * Ls if Ls_sum is None else Ls_sum + (1.0 / beta[v]) * Ls
    F, _, _ = eig1(Ls_sum, c, 0)
    return F


def update_s(A, F, J, Y, W, beta, mu, gamma):
    """对应 updateS.m：逐列解单纯形 QP。"""
    import osqp
    V = len(A)
    m = A[0].shape[1]
    S = [np.zeros((m, m)) for _ in range(V)]
    distF = sqdist(F, F)                               # (m, m)

    # 约束：0 <= x <= 1（box）且 1'x = 1（等式）→ 拼成 [I; 1']
    Acon = sp.vstack([sp.eye(m, format="csc"), sp.csc_matrix(np.ones((1, m)))])
    lcon = np.concatenate([np.zeros(m), np.array([1.0])])
    ucon = np.concatenate([np.ones(m), np.array([1.0])])
    for v in range(V):
        K = A[v].T @ W[v] @ W[v].T @ A[v]              # (m, m)
        Q = J[v] - Y[v] / mu
        H = mu * np.eye(m) + 2.0 * K / beta[v]
        H = (H + H.T) / 2.0
        P = sp.csc_matrix(H)
        prob = osqp.OSQP()
        prob.setup(P=P, q=np.zeros(m), A=Acon, l=lcon, u=ucon,
                   eps_abs=1e-4, eps_rel=1e-4, max_iter=1000,
                   verbose=False, warm_starting=True)
        tmpS = np.zeros((m, m))
        prev = np.asarray(S[v], dtype=np.float64)       # 上一轮的 S_v，作热启动
        for i in range(m):
            ff = distF[:, i] * gamma / beta[v] - 2.0 * K[i, :] / beta[v] - mu * Q[:, i]
            prob.update(q=ff)
            prob.warm_start(x=prev[:, i])
            res = prob.solve()
            tmpS[:, i] = res.x
        S[v] = tmpS
    return S


def update_j(S, Y, p, mu, lam):
    """对应 updateJ.m：张量 Schatten-p 近端。返回 (J, Y)。"""
    V = len(S)
    m = S[0].shape[0]
    sX = [m, m, V]
    S_tensor = np.stack(S, axis=2)
    Y_tensor = np.stack(Y, axis=2)
    s = S_tensor.reshape(-1, order="F")
    y = Y_tensor.reshape(-1, order="F")
    rho_scalar = lam / mu                          # 代码里 weight_vector=ones(V)，退化为标量
    myj = wshrink_obj_weight_lp(s + y / mu, rho_scalar, sX, 0, 3, p)
    J_tensor = myj.reshape(sX, order="F")
    return [J_tensor[:, :, v] for v in range(V)], [Y[v] for v in range(V)]


def update_y(Y, S, J, mu):
    """对应 updateY.m。"""
    return [Y[v] + mu * (S[v] - J[v]) for v in range(len(S))]


def _update_w_detail(X, A, Z, S, alpha, beta, r):
    """对应 updateW_detail.m。"""
    St = X @ X.T
    H2 = X @ Z @ A.T
    P = St - (H2 + H2.T) + A @ np.diag(Z.sum(axis=0)) @ A.T
    P = (P + P.T) / 2.0
    P = P / alpha
    Q1 = A @ S.T @ A.T
    Q = A @ A.T - (Q1 + Q1.T) + A @ S @ S.T @ A.T
    Q = (Q + Q.T) / 2.0
    Q = Q / beta
    T = P + Q
    T = (T + T.T) / 2.0
    H = np.linalg.lstsq(St, T, rcond=None)[0]
    r = int(min(r, St.shape[0]))
    W, _, _ = eig1(H, r, 0, 0)
    W = np.real(W)
    W = W @ np.diag(1.0 / np.sqrt(np.diag(W.T @ W) + EPS))
    return W


def update_w(X, A, Z, S, alpha, beta, r):
    """对应 updateW.m。"""
    return [_update_w_detail(X[v], A[v], Z[v], S[v], alpha[v], beta[v], r[v])
            for v in range(len(X))]


# --------------------------------------------------------------------------- #
# 加权平均 / CLR / 标签传播
# --------------------------------------------------------------------------- #
def weighted_average(data_cell, weights):
    """对应 weighted_average.m：按 1/weight 加权平均。"""
    final = (1.0 / weights[0]) * data_cell[0]
    for v in range(1, len(data_cell)):
        final = final + (1.0 / weights[v]) * data_cell[v]
    return final / (np.sum(1.0 / np.asarray(weights)) + EPS)


def clr(A0, c):
    """对应 CLR.m（约束拉普拉斯秩）。返回学习到的相似矩阵 A。"""
    NITER = 30
    zr = 1e-10
    lam = 0.1

    A0 = A0 - np.diag(np.diag(A0))
    num = A0.shape[0]
    A10 = (A0 + A0.T) / 2.0
    D10 = np.diag(A10.sum(axis=0))
    L0 = D10 - A10
    F0, _, evs = eig1(L0, num, 0)
    F = F0[:, :c]
    if evs[:c + 1].sum() < zr:
        raise ValueError(f"原图有超过 {c} 个连通分量")
    if evs[:c].sum() < zr:
        n_comp, _ = sp.csgraph.connected_components(sp.csr_matrix(A10), directed=False)
        return A0

    A = A0.copy()
    for _ in range(NITER):
        dist = sqdist(F, F)
        S = np.zeros((num, num))
        for i in range(num):
            a0 = A0[i, :]
            idxa0 = np.flatnonzero(a0 > 0)
            if idxa0.size == 0:
                idxa0 = np.arange(num)
            ai = a0[idxa0]
            di = dist[i, idxa0]
            ad = ai - 0.5 * lam * di
            S[i, idxa0] = eproj_simplex(ad)
        A = (S + S.T) / 2.0
        D = np.diag(A.sum(axis=0))
        L = D - A
        F_old = F
        F, _, ev = eig1(L, c, 0)
        fn1 = ev[:c].sum()
        fn2 = ev[:c + 1].sum()
        if fn1 > zr:
            lam = 2 * lam
        elif fn2 < zr:
            lam = lam / 2
            F = F_old
        else:
            break
    return A


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def run_etap(cubes, n_classes, num_pixel, p_value, lam, gamma, iter_num,
             proj_dims, t=1e2, dk=5, k=5, seed=8):
    """对应 demo.m + main.m。

    Parameters
    ----------
    cubes : list[(H, W, B_v)]  各视图立方体（float）
    n_classes : int            类别数 c
    num_pixel : int            超像素/锚点数 m
    p_value, lam, gamma        p 范数 / 张量正则 λ / 图正则 γ
    iter_num : int             迭代轮数
    proj_dims : list[int]      逐视图投影维度 r_v
    t, dk, k                   去噪参数 / 去噪近邻数 / 锚图近邻数

    Returns
    -------
    y_pred : (N,) int32        每像素标签（F 序）
    """
    V = len(cubes)
    m = int(num_pixel)
    c = int(n_classes)

    X, sp_labels = preprocess(cubes, t, dk, m)

    alpha = np.ones(V) / V
    beta = np.ones(V) / V

    A = init_a(X, sp_labels, m)
    Z = init_z(X, A, k, alpha)
    W = init_w(X, A, Z, proj_dims)

    rng = np.random.default_rng(seed)
    S = [rng.random((m, m)) for _ in range(V)]
    S = [s / s.sum(axis=0, keepdims=True) for s in S]
    J = [np.zeros((m, m)) for _ in range(V)]
    Y = [np.zeros((m, m)) for _ in range(V)]

    mu = 0.0001
    mu_max = 1e12
    eta = 1.1

    for _ in range(int(iter_num)):
        Z = update_z(X, A, k, W, alpha)
        A = update_a(X, Z, S, alpha, beta)
        alpha = update_alpha(W, X, A, Z)
        beta = update_beta(A, S, W)
        F = update_f(S, alpha, c)              # 官方代码此处传 alpha（见 main.m:71）
        S = update_s(A, F, J, Y, W, beta, mu, gamma)
        J, Y = update_j(S, Y, p_value, mu, lam)
        Y = update_y(Y, S, J, mu)
        W = update_w(X, A, Z, S, alpha, beta, proj_dims)
        mu = min(mu * eta, mu_max)

    Final_S = weighted_average(S, beta)
    Final_Z = weighted_average(Z, alpha)

    S0 = clr(Final_S.T, c)
    n_comp, U_label = sp.csgraph.connected_components(
        sp.csr_matrix(S0), directed=False)

    X_subLabel = np.argmax(Final_Z, axis=1)
    y_pred = U_label[X_subLabel].astype(np.int32)

    return y_pred


# --------------------------------------------------------------------------- #
# 联邦化主入口（视图划分）
# --------------------------------------------------------------------------- #
def run_etap_federated(cubes, n_classes, num_pixel, p_value, lam, gamma, iter_num,
                       proj_dims, t=1e2, dk=5, k=5, seed=8):
    """ETAP 联邦化版本（视图划分 / 纵向联邦，无损）。

    与 `run_etap` 唯一的差别是：把跨视图的聚合步骤（α、β、F、张量 t-SVD）
    标成「服务器」步骤，并在边界处统计通信量。计算本身逐函数复用，因此与
    集中式**逐位相同**。

    跨视图耦合（全部在锚点级 m×m，通信极小）：
      - α_v、β_v：每个客户端上传一个标量（trace / 残差范数），服务器归一化
      - F：每个客户端上传 Ls_v (m×m)，服务器加权求和 + 特征分解，广播 F (m×c)
      - 张量 Schatten-p：每个客户端上传 S_v (m×m)，服务器 t-SVD，广播 J_v (m×m)
      - 最终共识：上传 S_v (m×m) 与 Z_v (N×m)，服务器 CLR + 标签传播

    Returns
    -------
    y_pred : (N,) int32
    comm   : dict，通信量统计（bytes）
    """
    V = len(cubes)
    m = int(num_pixel)
    c = int(n_classes)
    N = int(cubes[0].shape[0] * cubes[0].shape[1])

    X, sp_labels = preprocess(cubes, t, dk, m)          # 客户端本地（原始数据不出门）

    alpha = np.ones(V) / V
    beta = np.ones(V) / V

    A = init_a(X, sp_labels, m)                          # 客户端本地
    Z = init_z(X, A, k, alpha)                           # 客户端本地
    W = init_w(X, A, Z, proj_dims)                       # 客户端本地

    rng = np.random.default_rng(seed)
    S = [rng.random((m, m)) for _ in range(V)]
    S = [s / s.sum(axis=0, keepdims=True) for s in S]
    J = [np.zeros((m, m)) for _ in range(V)]
    Y = [np.zeros((m, m)) for _ in range(V)]

    mu = 0.0001
    mu_max = 1e12
    eta = 1.1

    bmm = 8 * m * m          # 一个 m×m float64 矩阵的字节数
    bmc = 8 * m * c          # F (m×c)
    bNm = 8 * N * m          # Z_v (N×m)
    comm_up = 0
    comm_down = 0

    for _ in range(int(iter_num)):
        Z = update_z(X, A, k, W, alpha)                  # 客户端本地
        A = update_a(X, Z, S, alpha, beta)               # 客户端本地

        alpha = update_alpha(W, X, A, Z)                 # 服务器：上传 V 标量，广播 α
        comm_up += V * 8
        comm_down += V * 8

        beta = update_beta(A, S, W)                      # 服务器：上传 V 标量，广播 β
        comm_up += V * 8
        comm_down += V * 8

        F = update_f(S, alpha, c)                        # 服务器：上传 Ls_v，广播 F
        comm_up += V * bmm
        comm_down += V * bmc

        S = update_s(A, F, J, Y, W, beta, mu, gamma)     # 客户端本地（F 已广播）

        J, Y = update_j(S, Y, p_value, mu, lam)          # 服务器：上传 S_v，广播 J_v
        comm_up += V * bmm
        comm_down += V * bmm

        Y = update_y(Y, S, J, mu)                        # 客户端本地
        W = update_w(X, A, Z, S, alpha, beta, proj_dims)  # 客户端本地
        mu = min(mu * eta, mu_max)

    # 最终共识：上传 S_v 与 Z_v，服务器 CLR + 标签传播
    comm_up += V * (bmm + bNm)

    Final_S = weighted_average(S, beta)
    Final_Z = weighted_average(Z, alpha)
    S0 = clr(Final_S.T, c)
    n_comp, U_label = sp.csgraph.connected_components(
        sp.csr_matrix(S0), directed=False)
    X_subLabel = np.argmax(Final_Z, axis=1)
    y_pred = U_label[X_subLabel].astype(np.int32)

    comm = {
        "comm_up_bytes": int(comm_up),
        "comm_down_bytes": int(comm_down),
        "comm_bytes": int(comm_up + comm_down),
        "comm_rounds": int(iter_num),
        "comm_per_client_per_round": int(2 * 8 + 2 * bmm),
        "comm_final_z_bytes": int(V * bNm),
        "n_clients": int(V),
    }
    return y_pred, comm
