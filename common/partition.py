"""联邦划分层 —— 两种范式的统一接口。

范式语义
--------
**范式 B（视图划分 / 纵向联邦）**：客户端 m 持有 **第 m 个视图的全部像素**。
    这是本领域的主流设定（12 篇里有 10 篇如此），也是 FedRSMVC、CeFMC、
    EFIMVC、FIMCFG 等的原生设定。

**范式 A（区域划分 / 水平联邦）**：客户端 m 持有 **第 m 块地理区域的全部视图**。
    这是 Fed-MVKM 的原生设定，也是遥感里最现实的「不同机构持有不同地区」。

统一表示
--------
`Client.view_idx`  该客户端持有哪些视图
`Client.sample_idx` 该客户端持有哪些像素；`None` 表示全部像素

于是：
- 范式 B：`view_idx=[i]`, `sample_idx=None`
- 范式 A：`view_idx=[0..V-1]`, `sample_idx=<该区域像素>`

⚠️ 与 OSMAGC 的兼容性（重要）
----------------------------
OSMAGC 的空间正则需要在 **连通的 (H,W) 子栅格** 上构造差分算子。
因此：
- 范式 A 下**必须用保持空间连通性的切分**（网格 / 条带）；
  `random_partition` 会把栅格打散，OSMAGC 联邦化在它上面**无法构造 `construct_D`**。
- 范式 B 下每个客户端持有整幅栅格，`construct_D(H, W)` 直接可用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import RSDataset


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Client:
    client_id: int
    name: str
    view_idx: list[int]
    sample_idx: np.ndarray | None        # None = 全部像素
    meta: dict = field(default_factory=dict)

    def n_samples(self, n_total: int) -> int:
        return n_total if self.sample_idx is None else int(self.sample_idx.size)


@dataclass
class Partition:
    mode: str                            # 'view' | 'region' | 'random'
    shape: str                           # 人类可读描述，如 'view' / '2x2' / 'stripe4'
    clients: list[Client]
    n_views: int
    n_pixels: int
    grid: tuple[int, int]                # (H, W)
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.clients)

    def describe(self) -> str:
        head = (
            f"Partition(mode={self.mode}, shape={self.shape}, "
            f"clients={len(self.clients)}, grid={self.grid[0]}x{self.grid[1]})"
        )
        lines = [head]
        for c in self.clients:
            n = c.n_samples(self.n_pixels)
            views = ",".join(str(v) for v in c.view_idx)
            extra = ""
            if "class_counts" in c.meta:
                cc = c.meta["class_counts"]
                top = np.argsort(cc)[::-1][:3]
                extra = "  主导类: " + ", ".join(
                    f"c{k + 1}={100 * cc[k] / max(cc.sum(), 1):.0f}%" for k in top
                )
            lines.append(
                f"  [{c.client_id}] {c.name:<12} 视图=[{views}] 像素={n:>8,}{extra}"
            )
        if self.stats:
            lines.append(f"  stats: {self.stats}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 非 IID 度量
# --------------------------------------------------------------------------- #
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen–Shannon 散度（以 2 为底，取值 [0,1]）。0 表示分布相同。"""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-300))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _class_counts(gt_flat: np.ndarray, idx: np.ndarray | None, n_classes: int) -> np.ndarray:
    g = gt_flat if idx is None else gt_flat[idx]
    g = g[g > 0]
    return np.bincount(g, minlength=n_classes + 1)[1 : n_classes + 1].astype(np.float64)


def analyze_partition(
    ds: RSDataset, part: Partition, order: str = "F"
) -> Partition:
    """给划分加上非 IID 统计（逐客户端类别分布 / JS 散度 / 缺失类）。

    对范式 B 无意义（所有客户端样本相同），只统计样本量。
    """
    gt = ds.flatten_gt(order)
    K = ds.n_classes

    if part.mode == "view":
        part.stats = {
            "说明": "视图划分：所有客户端共享同一批像素，不产生样本级非IID",
            "每客户端像素数": ds.n_pixels,
        }
        return part

    global_counts = _class_counts(gt, None, K)
    global_p = global_counts / max(global_counts.sum(), 1)

    js_list, miss_list = [], []
    for c in part.clients:
        cc = _class_counts(gt, c.sample_idx, K)
        c.meta["class_counts"] = cc
        p = cc / max(cc.sum(), 1)
        js = _js_divergence(p, global_p)
        c.meta["js_vs_global"] = js
        c.meta["missing_classes"] = [
            int(k + 1) for k in range(K) if cc[k] == 0
        ]
        js_list.append(js)
        miss_list.append(len(c.meta["missing_classes"]))

    part.stats = {
        "全局类别占比": np.round(global_p, 4).tolist(),
        "JS散度(均值)": round(float(np.mean(js_list)), 4),
        "JS散度(最大)": round(float(np.max(js_list)), 4),
        "缺类的客户端数": int(sum(1 for m in miss_list if m > 0)),
        "各客户端像素数": [int(c.n_samples(ds.n_pixels)) for c in part.clients],
    }
    return part


# --------------------------------------------------------------------------- #
# 范式 B：视图划分
# --------------------------------------------------------------------------- #
def view_partition(ds: RSDataset) -> Partition:
    """每个视图 = 一个客户端（纵向联邦）。"""
    clients = [
        Client(
            client_id=i,
            name=f"view:{ds.view_names[i]}",
            view_idx=[i],
            sample_idx=None,
        )
        for i in range(ds.n_views)
    ]
    return analyze_partition(
        ds,
        Partition(
            mode="view",
            shape="view",
            clients=clients,
            n_views=ds.n_views,
            n_pixels=ds.n_pixels,
            grid=ds.grid_shape(),
        ),
    )


# --------------------------------------------------------------------------- #
# 范式 A：区域划分
# --------------------------------------------------------------------------- #
def region_grid_partition(ds: RSDataset, rows: int, cols: int, order: str = "F") -> Partition:
    """把栅格切成 rows×cols 个规则区块，每块 = 一个客户端。保持空间连通。"""
    H, W = ds.grid_shape()
    r_edges = np.linspace(0, H, rows + 1).astype(int)
    c_edges = np.linspace(0, W, cols + 1).astype(int)
    lab = np.zeros((H, W), dtype=np.int32)
    k = 0
    for i in range(rows):
        for j in range(cols):
            lab[r_edges[i] : r_edges[i + 1], c_edges[j] : c_edges[j + 1]] = k
            k += 1
    return _region_from_labels(ds, lab, f"{rows}x{cols}", order=order)


def region_stripe_partition(ds: RSDataset, n: int, axis: str = "col", order: str = "F") -> Partition:
    """切成 n 条条带（axis='col' 竖条带 / 'row' 横条带）。保持空间连通。"""
    H, W = ds.grid_shape()
    lab = np.zeros((H, W), dtype=np.int32)
    if axis == "col":
        edges = np.linspace(0, W, n + 1).astype(int)
        for j in range(n):
            lab[:, edges[j] : edges[j + 1]] = j
    elif axis == "row":
        edges = np.linspace(0, H, n + 1).astype(int)
        for j in range(n):
            lab[edges[j] : edges[j + 1], :] = j
    else:
        raise ValueError(f"axis 只能是 'col' 或 'row'，收到 {axis!r}")
    return _region_from_labels(ds, lab, f"stripe{n}{'V' if axis == 'col' else 'H'}", order=order)


def _region_from_labels(ds: RSDataset, lab: np.ndarray, shape_name: str, order: str = "F") -> Partition:
    """由 (H,W) 的区块标签图构造区域划分。"""
    flat_lab = lab.reshape(-1, order=order)
    clients = []
    for k in range(int(flat_lab.max()) + 1):
        idx = np.flatnonzero(flat_lab == k)
        clients.append(
            Client(
                client_id=k,
                name=f"region{k}",
                view_idx=list(range(ds.n_views)),
                sample_idx=idx,
                meta={"bbox_rc": _bbox(lab == k)},
            )
        )

    part = Partition(
        mode="region",
        shape=shape_name,
        clients=clients,
        n_views=ds.n_views,
        n_pixels=ds.n_pixels,
        grid=ds.grid_shape(),
    )
    _check_covers_all(part, ds.n_pixels)
    return analyze_partition(ds, part, order=order)


def _bbox(mask2d: np.ndarray) -> list[int]:
    rows = np.flatnonzero(mask2d.any(axis=1))
    cols_ = np.flatnonzero(mask2d.any(axis=0))
    return [int(rows[0]), int(rows[-1]), int(cols_[0]), int(cols_[-1])]


def random_partition(
    ds: RSDataset, m: int, seed: int = 0, alpha: float | None = None, order: str = "F"
) -> Partition:
    """随机样本划分（打散栅格）。

    `alpha=None` 为均匀随机（IID）；`alpha` 给定则按类别做 Dirichlet 非 IID 分配。

    ⚠️ 该划分**破坏空间连通性**，OSMAGC 联邦化在其上无法构造 `construct_D(H,W)`。
    它在本平台里的角色是**对照组**：用来量化「空间结构本身值多少」。
    """
    gt = ds.flatten_gt(order)
    N = gt.size
    rng = np.random.default_rng(seed)

    assign = np.empty(N, dtype=np.int32)
    if alpha is None:
        assign[:] = rng.integers(0, m, N)
    else:
        for k in range(1, ds.n_classes + 1):
            idx = np.flatnonzero(gt == k)
            if idx.size == 0:
                continue
            prop = rng.dirichlet(np.full(m, alpha))
            assign[idx] = rng.choice(m, size=idx.size, p=prop)
        # 背景像素（gt==0）均匀分配
        bg = np.flatnonzero(gt == 0)
        if bg.size:
            assign[bg] = rng.integers(0, m, bg.size)

    clients = [
        Client(
            client_id=k,
            name=f"rand{k}",
            view_idx=list(range(ds.n_views)),
            sample_idx=np.flatnonzero(assign == k),
        )
        for k in range(m)
    ]
    part = Partition(
        mode="random",
        shape=f"random{m}" + ("" if alpha is None else f"_dir{alpha}"),
        clients=clients,
        n_views=ds.n_views,
        n_pixels=N,
        grid=ds.grid_shape(),
    )
    _check_covers_all(part, N)
    return analyze_partition(ds, part, order=order)


def _check_covers_all(part: Partition, n_total: int) -> None:
    """校验划分是「恰好一次」的覆盖，防止静默丢样本/重复样本。"""
    if part.mode == "view":
        return
    seen = np.zeros(n_total, dtype=np.int8)
    for c in part.clients:
        idx = c.sample_idx
        if idx is None:
            raise ValueError(f"region/random 划分下客户端 {c.client_id} 未指定 sample_idx")
        if seen[idx].any():
            raise ValueError(f"客户端 {c.client_id} 与其他客户端样本重叠")
        seen[idx] = 1
    if not seen.all():
        missing = int((seen == 0).sum())
        raise ValueError(f"划分未覆盖全部像素，缺失 {missing} 个")
