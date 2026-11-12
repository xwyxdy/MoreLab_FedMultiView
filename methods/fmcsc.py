"""FMCSC（Federated Multi-view Clustering with Client Heterogeneity）适配器。

来源：https://github.com/5Martina5/FMCSC
接口：run_fmcsc(ds, seed=0, **kwargs) -> (labels (N,) int32, detail)

═══════════════════════════════════════════════════════════════════════════
 一、这篇论文的客户端构成，以及本适配器取的是哪一种
═══════════════════════════════════════════════════════════════════════════

原论文是「单/多混合」（范式 C）：
    M 个多视图客户端（每人持全部 V 个视图）+ S 个单视图客户端（每人只持 1 个视图）
    由 `--M_S` 控制比例（论文默认 2:1 / 1:1 / 1:2）
    由此产生三种 gap：client gap（谁有哪个视图）、view gap（缺视图）、model drift

本脚手架是「每人全视图」（horizontal），所以这里取 **S = 0**：
    全部 N_c 个客户端都是多视图客户端，每个客户端持 V 个视图的全部像素子集。
    这不是"改方法"，而是原方法在 M_S = ∞ 时的退化配置：
      - client gap 消失（人人视图相同 → 聚合只有一个分组，glob_models 只有 1 个）
      - view gap 消失（每客户端的 num_view == V → local_full_train 走"全视图"分支）
      - model drift 仍在（每人只见过自己那部分像素）
    原代码里 `missing_rate = missing_mapping.get(args.M_S)`，M_S 越大缺失率越低，
    S=0 对应 missing_rate=0，正好是脚手架的"完整数据"设定。

═══════════════════════════════════════════════════════════════════════════
 二、原实现的四段流程（本文件逐段对应，行号指 FMCSC-main/main.py）
═══════════════════════════════════════════════════════════════════════════

  ① 共识预训练（86-122 行 `pretrain`）
     客户端 0 从零训练自编码器（MSE 重建）；其余客户端**继承 0 号客户端训完的权重**
     再各自训练（代码 462-465 行：`local_models[nu+1] = local_models[0]`，
     注意这是**串行**，每次赋值都指向 0 号，所以所有人最终都从"0 号训完的状态"起步）。
     训完在每个客户端上跑 k-means 填 `model.centroids`（117-122 行）。

  ② 首次聚合（467-470 行）
     全部客户端等权平均 → 一个全局模型；再按"视图组合"分组（`aggregate_models1`）。
     S=0 时所有客户端视图组合相同 → 只有 1 个分组。

  ③ 5 轮通信（473-497 行）
     每个客户端先 `local_train`（126-151 行）：损失 = Σ_v InfoNCE(r_v, h) + Σ_v MSE(x_v, xr_v)，
     其中 InfoNCE 是"最大化 I(r_v; h)"的对比式代理（loss.py:28-44 `forward_feature`）。
     再 `local_full_train`（153-197 行）：让**全局模型的副本**在本地数据上学
        Σ_v MSE(x_v, xr_v) + MSE(h, h_ref)，h_ref 来自本地模型（教师信号）；
     然后算 MI 权重 = Σ_v MI(h, r_v)（metric.py 的 10-bin 二维直方图）。

  ④ 服务器加权聚合（365-380 行 `aggregate_models2`，即论文 Eq.10）
     按各客户端报的 MI 权重做加权平均。**权重未归一化**（直接用原始 MI 值，
     再由 sum 归一）——这是论文与代码不一致之一，此处照代码。

  最后读标签：`valid()`（274-310 行）——每个客户端用自己的（分组的）全局模型算 h，
  第一个客户端用 k-means 得到初始中心，后续客户端 `init=init_center` 复用同一组中心，
  各客户端的标签拼起来就是全体像素的标签。

═══════════════════════════════════════════════════════════════════════════
 三、与原实现的三处差异（都是为了能在脚手架上跑，且已记在 detail 里）
═══════════════════════════════════════════════════════════════════════════

  (1) **轮数与 epoch 数被压缩**：原论文 MNIST-USPS 用 mse_epochs=250、main_epochs=25
      （`local_full_train` 里还硬编码 50 轮）；这里默认 pretrain=30 / main=5 / full=5。
      原因：Trento 有 9.96 万像素，按原轮数单次要跑十几小时。
      **所以这个配置下的数字不能当"1:1 复现"，只能当"同数据同口径的对照"。**

  (2) **特征做了逐视图 z-score**：原仓库 `dataloader.py` 直接把原始特征喂进网络，
      遥感特征（HS 反射率、LiDAR 高程）量级差 3 个数量级，不归一自编码器会只拟合大尺度那一路。

  (3) `centroids` 这个 Parameter **建了但没参与任何损失**（原仓库也是如此，
      `main.py:117-122` 填完后再无引用；聚类最终走 sklearn KMeans）。
      这里保留该字段以保持结构一致，但不参与训练——**不是漏写**。

  ⚠️ 原仓库 `loss.py:14` 的 mask 是按 `args.batch_size` 预先建好的，
      最后一个 batch 不满时会形状不匹配；这里改成按实际 batch 建 mask（等价，只是不再依赖整除）。
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

try:                                        # 脚手架统一把 BLAS 线程固定成 4
    from common.threads import THREADS
except Exception:                           # noqa: BLE001
    THREADS = 4


# ═══════════════════════════════════════════════════════════════════════════
#  网络：与 FMCSC-main/my_network.py 同构（500 / 500 / 2000 / feature_dim）
# ═══════════════════════════════════════════════════════════════════════════
class _Encoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, 2000), nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class _Decoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000), nn.ReLU(),
            nn.Linear(2000, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class _FeatureR(nn.Module):
    """feature_R_module：把每个视图的 z 投到 20 维，供 MI / InfoNCE 用。"""

    def __init__(self, feature_dim: int, r_dim: int = 20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, r_dim),
        )

    def forward(self, z):
        return self.net(z)


class _Net(nn.Module):
    """一个客户端的模型：每个视图一套 enc/dec/r + 一个跨视图融合头。"""

    def __init__(self, in_dims, feature_dim: int, class_num: int):
        super().__init__()
        self.V = len(in_dims)
        self.encoders = nn.ModuleList([_Encoder(d, feature_dim) for d in in_dims])
        self.decoders = nn.ModuleList([_Decoder(d, feature_dim) for d in in_dims])
        self.feature_Rs = nn.ModuleList([_FeatureR(feature_dim) for _ in in_dims])
        # 与原作者一致：建了但从未使用（见文件头 三(3)）
        self.centroids = nn.Parameter(torch.zeros(class_num, 20))
        self.feature_H_module = nn.Sequential(
            nn.Linear(self.V * feature_dim, 256), nn.ReLU(), nn.Linear(256, 20),
        )

    def forward(self, xs):
        zs, xrs, rs = [], [], []
        for v in range(self.V):
            z = self.encoders[v](xs[v])
            xrs.append(self.decoders[v](z))
            rs.append(self.feature_Rs[v](z))
            zs.append(z)
        h = F.normalize(self.feature_H_module(torch.cat(zs, dim=1)), dim=1)
        return xrs, zs, h, rs


# ═══════════════════════════════════════════════════════════════════════════
#  两个损失与一个权重（照原实现）
# ═══════════════════════════════════════════════════════════════════════════
def _info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    """loss.py `forward_feature` 的等价实现：同下标 (a_i, b_i) 为正样本，其余为负。"""
    n = a.shape[0]
    h = torch.cat([a, b], dim=0)
    sim = (h @ h.t()) / tau
    pos = torch.cat([torch.diagonal(sim, n), torch.diagonal(sim, -n)])
    m = torch.ones((2 * n, 2 * n), dtype=torch.bool, device=a.device)
    m.fill_diagonal_(False)
    for i in range(n):                      # 跨半区的正样本对不参与负样本
        m[i, n + i] = False
        m[n + i, i] = False
    neg = sim[m].reshape(2 * n, -1)
    logits = torch.cat([pos.reshape(-1, 1), neg], dim=1)
    return F.cross_entropy(logits, torch.zeros(2 * n, dtype=torch.long,
                                               device=a.device), reduction="sum") / (2 * n)


def _mutual_information(x: torch.Tensor, y: torch.Tensor, bins: int = 10) -> float:
    """metric.py `mutual_information`：把两个矩阵整体展平，做 10-bin 二维直方图。"""
    a = x.detach().flatten().cpu().numpy()
    b = y.detach().flatten().cpu().numpy()
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    pj = hist / hist.sum()
    p1 = hist.sum(axis=1) / hist.sum()
    p2 = hist.sum(axis=0) / hist.sum()
    pj[pj == 0] = 1e-10
    p1[p1 == 0] = 1e-10
    p2[p2 == 0] = 1e-10
    return float(np.sum(pj * np.log2(pj / np.outer(p1, p2))))


def _weighted_mean(models, weights) -> dict:
    """aggregate_models(weights)：按权重做 state_dict 加权平均（论文 Eq.10）。"""
    ws = np.asarray(weights, dtype=np.float64)
    if ws.sum() <= 1e-12:                  # 全零 MI 时退化为等权，避免聚合出全零模型
        ws = np.ones_like(ws)
    ws = ws / ws.sum()
    out = copy.deepcopy(models[0].state_dict())
    for k in out:
        out[k] = torch.zeros_like(out[k])
    for w, m in zip(ws, models):
        sd = m.state_dict()
        for k in out:
            out[k] += sd[k] * float(w)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════════════════
def run_fmcsc(ds, seed: int = 0, num_clients: int = 5, feature_dim: int = 20,
              pretrain_epochs: int = 30, main_epochs: int = 5,
              full_epochs: int = 5, rounds: int = 5, batch_size: int = 512,
              lr: float = 3e-4, temperature_f: float = 0.5,
              verbose: bool = False, **_):
    """FMCSC（S=0 全多视图）跑一次，返回全体像素的簇标签。

    Parameters
    ----------
    num_clients : 客户端数（每个都持全部视图，按像素 iid 切分，对应 Dirichlet α=9999）
    pretrain_epochs / main_epochs / full_epochs / rounds : 见文件头 三(1)，均小于论文原值
    """
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(THREADS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    V, N, K = ds.n_views, ds.n_pixels, ds.n_classes
    in_dims = ds.view_dims()

    # ---- 取数据：F 序展平 + 逐视图 z-score（见文件头 三(2)）----
    X_np = []
    for v in range(V):
        a = np.asarray(ds.flatten_view(v, "F"), dtype=np.float32)
        mu, sd = a.mean(0, keepdims=True), a.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        X_np.append(torch.from_numpy((a - mu) / sd))
    X = [x.to(device) for x in X_np]

    # ---- 客户端划分：iid（Dirichlet α=9999 ≈ 均匀随机），与论文默认一致 ----
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    parts = np.array_split(perm, num_clients)
    n_client = len(parts)

    models = [_Net(in_dims, feature_dim, K).to(device) for _ in range(n_client)]
    params = sum(p.numel() for p in models[0].parameters())
    model_bytes = params * 4

    def batches(idx: np.ndarray, bs: int):
        for s in range(0, len(idx), bs):
            yield idx[s:s + bs]

    # ═════ ① 共识预训练：0 号从零训，其余继承 0 号权重再各自训（串行，同原代码）═════
    for c in range(n_client):
        if c > 0:
            models[c].load_state_dict(copy.deepcopy(models[0].state_dict()))
        m = models[c]
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        mse = nn.MSELoss()
        m.train()
        for ep in range(pretrain_epochs):
            for bidx in batches(parts[c], batch_size):
                xs = [x[bidx] for x in X]
                opt.zero_grad(set_to_none=True)
                xrs, _, _, _ = m(xs)
                loss = sum(mse(xs[v], xrs[v]) for v in range(V))
                loss.backward()
                opt.step()
        # 用 k-means 填 centroids（117-122 行；该参数之后不再被使用，这里保留以对齐结构）
        m.eval()
        with torch.no_grad():
            hs = torch.cat([m([x[bidx] for x in X])[2]
                            for bidx in batches(parts[c], 4096)], 0)
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(hs.cpu().numpy())
        m.centroids.data = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)

    # ═════ ② 首次聚合 + 分组（S=0 → 只有 1 个分组）═════
    glob_sd = _weighted_mean(models, np.ones(n_client))
    gmodel = _Net(in_dims, feature_dim, K).to(device)
    gmodel.load_state_dict(glob_sd)

    comm_bytes = 0
    round_log = []

    # ═════ ③④ 5 轮通信 + 服务器按 MI 加权聚合 ═════
    for rd in range(rounds):
        local_models = []
        mi_weights = []
        for c in range(n_client):
            # ---- local_train：本地模型从全局模型起步，跑 main_epochs ----
            loc = _Net(in_dims, feature_dim, K).to(device)
            loc.load_state_dict(copy.deepcopy(glob_sd))
            opt = torch.optim.Adam(loc.parameters(), lr=lr)
            mse = nn.MSELoss()
            loc.train()
            for ep in range(main_epochs):
                for bidx in batches(parts[c], batch_size):
                    xs = [x[bidx] for x in X]
                    opt.zero_grad(set_to_none=True)
                    xrs, zs, h, rs = loc(xs)
                    loss = sum(mse(xs[v], xrs[v]) for v in range(V))
                    loss = loss + sum(_info_nce(rs[v], h, temperature_f) for v in range(V))
                    loss.backward()
                    opt.step()

            # h_ref：本地模型在**全部本地数据**上的融合特征（教师信号）
            loc.eval()
            h_ref = torch.cat([loc([x[bidx] for x in X])[2]
                               for bidx in batches(parts[c], 4096)], 0).detach()

            # ---- local_full_train：全局副本在本地数据上学重建 + 对齐 h_ref ----
            g = _Net(in_dims, feature_dim, K).to(device)
            g.load_state_dict(copy.deepcopy(glob_sd))
            opt = torch.optim.Adam(g.parameters(), lr=lr)
            g.train()
            for ep in range(full_epochs):
                for s in range(0, len(parts[c]), batch_size):
                    e = min(s + batch_size, len(parts[c]))
                    xs = [x[parts[c][s:e]] for x in X]
                    opt.zero_grad(set_to_none=True)
                    xrs, _, h, _ = g(xs)
                    loss = sum(mse(xs[v], xrs[v]) for v in range(V))
                    loss = loss + mse(h, h_ref[s:e])      # 教师信号，行号对齐
                    loss.backward()
                    opt.step()

            # ---- MI 权重（全视图分支：Σ_v MI(h, r_v)）----
            g.eval()
            with torch.no_grad():
                h_parts, r_parts = [], [[] for _ in range(V)]
                for bidx in batches(parts[c], 4096):
                    xs = [x[bidx] for x in X]
                    _, _, h, rs = g(xs)
                    h_parts.append(h)
                    for v in range(V):
                        r_parts[v].append(rs[v])
                h_all = torch.cat(h_parts, 0)
                r_all = [torch.cat(r_parts[v], 0) for v in range(V)]
            mi = sum(_mutual_information(h_all, r_all[v]) for v in range(V))
            mi_weights.append(mi)
            local_models.append(g)

        glob_sd = _weighted_mean(local_models, np.array(mi_weights))
        gmodel.load_state_dict(glob_sd)
        comm_bytes += n_client * 2 * model_bytes      # 上传模型 + 下发全局模型
        round_log.append([round(float(w), 4) for w in mi_weights])
        if verbose:
            el = time.time() - t0
            print(f"  [fmcsc] round {rd + 1}/{rounds} "
                  f"mi={np.round(mi_weights, 3).tolist()} t={el:.0f}s", flush=True)

    # ═════ 读标签：每客户端 k-means，后续客户端复用首个客户端的中心（valid()）═════
    gmodel.eval()
    labels = np.zeros(N, dtype=np.int32)
    init_centers = None
    for c in range(n_client):
        with torch.no_grad():
            hs = torch.cat([gmodel([x[bidx] for x in X])[2]
                            for bidx in batches(parts[c], 4096)], 0).cpu().numpy()
        if init_centers is None:
            km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(hs)
            init_centers = km.cluster_centers_
        else:
            km = KMeans(n_clusters=K, init=init_centers.copy(), n_init=1).fit(hs)
        labels[parts[c]] = km.labels_.astype(np.int32)

    el = time.time() - t0
    return labels, {
        "runtime_s": round(el, 1),
        "comm_bytes": int(comm_bytes),
        "model_params": int(params),
        "model_MB": round(model_bytes / 2 ** 20, 1),
        "rounds": rounds,
        "clients": n_client,
        "client_mode": "S=0 全多视图（M_S=∞ 退化配置）",
        "feature_dim": feature_dim,
        "epochs": f"pretrain={pretrain_epochs}/main={main_epochs}/full={full_epochs}",
        "mi_weights_per_round": round_log,
        "备注": "epoch 数小于论文原值（250/25/50），数字为同数据同口径对照，非 1:1 复现",
    }
