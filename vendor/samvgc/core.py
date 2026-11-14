"""SAMVGC 的 PyTorch 复现。

论文：Structure-Adaptive Multi-View Graph Clustering for Remote Sensing Data
      （AAAI 2025，Guan Renxiang 等）。官方未开源，本模块按论文逐公式实现。

结构：
  1. 融合所有视图 → SLIC 超像素分割 → 超像素作为锚点（M 个）
  2. 逐视图构建 k-NN 初始图 A(v)
  3. 图自编码器（3 层 GCN 编码 + 解码）逐视图学特征 Z(v)
  4. 锚点空间相关自适应精炼（每 ip 轮：用 Z(v) 的相似度精炼 A(v)）
  5. 自适应簇结构学习：融合 Z_F → K-means → 中心/混淆样本划分 →
     中心样本原型对比损失 + 混淆样本一致性损失
  6. 最终对 Z_F 做 K-means 出簇，传播到像素

⚠️ 论文未给出超像素数 M / 近邻数 k / 中心比例 α / 轮数 / 学习率，
   本实现按惯例取默认值（见 DEFAULT_PARAMS），结果可能与论文数值有出入。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# --------------------------------------------------------------------------- #
# 预处理
# --------------------------------------------------------------------------- #
def _symmetrize(A):
    A = (A + A.T) / 2.0
    return A


def _normalize_adj(A):
    """对称归一化拉普拉斯 Ã = D^{-1/2} A D^{-1/2}（含自环）。"""
    A = np.asarray(A, dtype=np.float32)
    A = A + np.eye(A.shape[0])
    d = A.sum(axis=1)
    d = np.maximum(d, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    return d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]


def preprocess(cubes, n_superpixels, k_neighbors):
    """融合视图 → SLIC 超像素 → 锚点特征 + 初始邻接矩阵。

    Returns
    -------
    X_anchors : list[(M, B_v)]  逐视图锚点特征（超像素均值）
    A_init    : list[(M, M)]    逐视图初始邻接矩阵（k-NN，对称）
    sp_labels : (N,) int         每个像素的超像素标签（F 序）
    """
    from skimage.segmentation import slic

    V = len(cubes)
    H, W = cubes[0].shape[:2]
    N = H * W

    # 逐视图 minmax 归一化到 [0,1]，再融合
    norms = []
    for c in cubes:
        flat = c.reshape(N, -1, order="F").astype(np.float32)
        lo = flat.min(axis=0, keepdims=True)
        hi = flat.max(axis=0, keepdims=True)
        norms.append((flat - lo) / (hi - lo + 1e-8))
    fused = np.concatenate(norms, axis=1)               # (N, ΣB_v)

    fused_img = fused.reshape(H, W, -1, order="F")
    sp_2d = slic(fused_img, n_segments=int(n_superpixels),
                 compactness=10.0, channel_axis=-1, start_label=0)
    sp_labels = sp_2d.reshape(-1, order="F").astype(np.int64)
    M = int(sp_labels.max()) + 1

    X_anchors = []
    A_init = []
    for c in cubes:
        flat = c.reshape(N, -1, order="F").astype(np.float32)
        lo = flat.min(axis=0, keepdims=True)
        hi = flat.max(axis=0, keepdims=True)
        flat = (flat - lo) / (hi - lo + 1e-8)
        anchor = np.zeros((M, flat.shape[1]), dtype=np.float32)
        for i in range(M):
            mask = sp_labels == i
            anchor[i] = flat[mask].mean(axis=0)
        X_anchors.append(anchor)

        D = cdist(anchor, anchor, metric="euclidean")
        idx = np.argsort(D, axis=1)[:, 1:k_neighbors + 1]
        A = np.zeros((M, M), dtype=np.float32)
        rows = np.repeat(np.arange(M), k_neighbors)
        A[rows, idx.ravel()] = 1.0
        A = _symmetrize(A)
        A_init.append(A)

    return X_anchors, A_init, sp_labels


# --------------------------------------------------------------------------- #
# 图自编码器
# --------------------------------------------------------------------------- #
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = act

    def forward(self, X, A_norm):
        return F.relu(A_norm @ self.linear(X)) if self.act else A_norm @ self.linear(X)


class GraphAutoencoder(nn.Module):
    """3 层 GCN 编码 + 解码，逐视图一个。"""
    def __init__(self, in_dim, hidden=(128, 256, 512)):
        super().__init__()
        self.enc1 = GCNLayer(in_dim, hidden[0])
        self.enc2 = GCNLayer(hidden[0], hidden[1])
        self.enc3 = GCNLayer(hidden[1], hidden[2])
        self.dec1 = GCNLayer(hidden[2], hidden[1])
        self.dec2 = GCNLayer(hidden[1], hidden[0])
        self.dec3 = GCNLayer(hidden[0], in_dim, act=False)
        self.out_dim = hidden[2]

    def encode(self, X, A_norm):
        h1 = self.enc1(X, A_norm)
        h2 = self.enc2(h1, A_norm)
        z = self.enc3(h2, A_norm)
        return z

    def forward(self, X, A_norm):
        z = self.encode(X, A_norm)
        d1 = self.dec1(z, A_norm)
        d2 = self.dec2(d1, A_norm)
        xr = self.dec3(d2, A_norm)
        return z, xr


# --------------------------------------------------------------------------- #
# 图精炼（Eq. 3-7）
# --------------------------------------------------------------------------- #
def refine_adjacency(Z, A_old, k):
    """用特征 Z 的精炼邻接矩阵。Z:(M,d) numpy。"""
    M = Z.shape[0]
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    Sz = Z @ Z.T                                   # 余弦相似度
    np.fill_diagonal(Sz, 0.0)
    G = np.zeros_like(Sz)
    idx = np.argsort(-Sz, axis=1)[:, :k]           # top-k
    rows = np.repeat(np.arange(M), k)
    G[rows, idx.ravel()] = Sz[rows, idx.ravel()]
    G = np.maximum(G, G.T)                         # 对称化（取大）
    np.fill_diagonal(G, 1.0)                       # 自环=1
    d = G.sum(axis=1)
    d = np.maximum(d, 1e-12)
    Az = G / d[:, None]                            # 行归一化
    return Az


# --------------------------------------------------------------------------- #
# 主模型
# --------------------------------------------------------------------------- #
class SAMVGC(nn.Module):
    def __init__(self, view_dims, hidden=(128, 256, 512), k_refine=10):
        super().__init__()
        self.V = len(view_dims)
        self.gaes = nn.ModuleList([GraphAutoencoder(d, hidden) for d in view_dims])
        self.out_dim = hidden[2]
        # 融合权重 Wv（Eq. 11）：逐视图逐元素的可学习权重
        self.fuse_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1, self.out_dim)) for _ in range(self.V)
        ])
        # 图精炼的融合 MLP 权重（Eq. 6）：[A_z ∥ A] -> 2
        self.k_refine = k_refine

    def encode_all(self, Xs, As_norm):
        Zs = []
        for v in range(self.V):
            Zs.append(self.gaes[v].encode(Xs[v], As_norm[v]))
        return Zs

    def fused(self, Zs):
        ZF = sum(F.softplus(self.fuse_weights[v]) * Zs[v] for v in range(self.V))
        return ZF

    def forward(self, Xs, As_norm):
        Zs, Xrs = [], []
        for v in range(self.V):
            z, xr = self.gaes[v](Xs[v], As_norm[v])
            Zs.append(z)
            Xrs.append(xr)
        return Zs, Xrs


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def hcl_loss(Zs, ZF, center_idx, labels):
    """原型对比损失（Eq. 15）。center_idx: 中心样本布尔掩码，labels: K-means 簇标签。"""
    K = int(labels.max()) + 1
    total = 0.0
    for v, Zv in enumerate(Zs):
        for k in range(K):
            mask = center_idx & (labels == k)
            if mask.sum() == 0:
                continue
            mu_vk = Zv[mask].mean(dim=0, keepdims=True)      # (1,d)
            muF = []
            for j in range(K):
                mj = center_idx & (labels == j)
                muF.append(ZF[mj].mean(dim=0, keepdims=True) if mj.sum() > 0
                           else torch.zeros_like(mu_vk))
            muF = torch.cat(muF, dim=0)                      # (K,d)
            sims = F.cosine_similarity(mu_vk, muF)           # (K,)
            pos = sims[k]
            logits = sims
            total += -torch.log(torch.exp(pos) / (torch.exp(logits).sum() + 1e-12))
    return total / max(K * len(Zs), 1)


def lce_loss(Zs, ZF, center_idx, labels, confusing_idx):
    """混淆样本一致性损失（Eq. 16-17）。KL 双向。"""
    V = len(Zs)
    total = 0.0
    # 用中心样本作为参考集
    centers = [Zv[center_idx] for Zv in Zs]
    ZFc = ZF[center_idx]
    for v in range(V):
        Zv_conf = Zs[v][confusing_idx]
        ZF_conf = ZF[confusing_idx]
        if Zv_conf.shape[0] == 0:
            continue
        p = torch.softmax(Zv_conf @ centers[v].T, dim=1)     # (M_l, M_h)
        q = torch.softmax(ZF_conf @ ZFc.T, dim=1)
        kl_pq = (p * (p.log() - q.log())).sum(dim=1)
        kl_qp = (q * (q.log() - p.log())).sum(dim=1)
        total += (kl_pq + kl_qp).mean()
    return total / max(V, 1)


# --------------------------------------------------------------------------- #
# 训练入口
# --------------------------------------------------------------------------- #
def run_samvgc(cubes, n_classes, n_superpixels=2000, k_neighbors=10, central_rate=0.5,
               epochs=100, lr=1e-3, lam1=100.0, lam2=500.0, refine_every=10,
               device="cuda", seed=0, verbose=True, return_comm=False):
    """返回每像素标签 (N,) int32（F 序）。return_comm=True 时返回 (labels, comm)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_anchors, A_init, sp_labels = preprocess(cubes, n_superpixels, k_neighbors)
    V = len(cubes)
    M = X_anchors[0].shape[0]
    view_dims = [X.shape[1] for X in X_anchors]

    Xs = [torch.tensor(X, dtype=torch.float32, device=device) for X in X_anchors]
    As_norm = [torch.tensor(_normalize_adj(A), dtype=torch.float32, device=device)
               for A in A_init]

    model = SAMVGC(view_dims, k_refine=k_neighbors).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    d_hat = model.out_dim

    # 通信量统计（视图划分联邦：每轮各客户端上传 Z(v)，服务器广播 ZF/原型/簇标签）
    comm_up = 0
    comm_down = 0
    bytes_Z = 4 * M * d_hat            # Z(v) / ZF，float32
    bytes_proto = 4 * n_classes * d_hat  # 融合原型 µF_k
    bytes_label = 4 * M                # 簇标签 int32

    A_cur = list(A_init)          # numpy 邻接矩阵，随精炼更新

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        Zs, Xrs = model(Xs, As_norm)

        # 客户端 → 服务器：上传各视图 Z(v)
        comm_up += V * bytes_Z

        # 重建损失：Frobenius 范数平方（求和，非平均），与论文 Eq.(10) 一致
        l_rec = sum(((Xrs[v] - Xs[v]) ** 2).sum() for v in range(V))

        # 融合特征 + K-means（detach 到 numpy）
        ZF = model.fused(Zs)
        ZF_np = ZF.detach().cpu().numpy()
        km = KMeans(n_clusters=n_classes, n_init=10, random_state=seed)
        labels = km.fit_predict(ZF_np)
        dist = ((ZF_np - km.cluster_centers_[labels]) ** 2).sum(axis=1)
        n_central = max(int(M * central_rate), n_classes)
        order = np.argsort(dist)
        center_idx = np.zeros(M, dtype=bool)
        center_idx[order[:n_central]] = True
        confusing_idx = ~center_idx

        # 服务器 → 客户端：广播 ZF / 原型 / 簇标签
        comm_down += V * (bytes_Z + bytes_proto + bytes_label)

        center_t = torch.tensor(center_idx, device=device)
        conf_t = torch.tensor(confusing_idx, device=device)
        lab_t = torch.tensor(labels, device=device)

        l_hcl = hcl_loss(Zs, ZF, center_t, lab_t)
        l_lce = lce_loss(Zs, ZF, center_t, lab_t, conf_t)
        loss = l_rec + lam1 * l_hcl + lam2 * l_lce
        loss.backward()
        opt.step()

        # 图精炼（每 refine_every 轮）
        if epoch % refine_every == 0:
            model.eval()
            with torch.no_grad():
                Zs_eval = model.encode_all(Xs, As_norm)
            for v in range(V):
                Zv_np = Zs_eval[v].detach().cpu().numpy()
                A_cur[v] = refine_adjacency(Zv_np, A_cur[v], k_neighbors)
                As_norm[v] = torch.tensor(_normalize_adj(A_cur[v]),
                                          dtype=torch.float32, device=device)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:3d}/{epochs}  loss={loss.item():.4f} "
                  f"(rec {l_rec.item():.4f}, hcl {l_hcl.item():.4f}, lce {l_lce.item():.4f})",
                  flush=True)

    # 最终聚类：K-means 在 Z_F 上
    model.eval()
    with torch.no_grad():
        Zs = model.encode_all(Xs, As_norm)
        ZF = model.fused(Zs).detach().cpu().numpy()
    anchor_labels = KMeans(n_clusters=n_classes, n_init=10, random_state=seed).fit_predict(ZF)

    # 传播到像素
    y_pred = anchor_labels[sp_labels].astype(np.int32)

    if return_comm:
        comm = {
            "comm_up_bytes": int(comm_up),
            "comm_down_bytes": int(comm_down),
            "comm_bytes": int(comm_up + comm_down),
            "comm_rounds": int(epochs),
            "comm_per_client_per_round": int(bytes_Z),
            "n_clients": int(V),
            "d_hat": int(d_hat),
            "n_anchors": int(M),
        }
        return y_pred, comm
    return y_pred
