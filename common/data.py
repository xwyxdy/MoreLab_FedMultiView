"""统一遥感多视图数据集加载层。

职责
----
把各项目里格式不一的 `.mat` 立方体，统一成同一个内存表示：
    cubes : [(H, W, B_v)] 逐视图的立方体
    gt    : (H, W)  int32，0 表示背景
    coords: (N, 2) 像素的行列坐标

⚠️ 关键约定：**像素展平一律用 Fortran（列优先）序**。
    OSMAGC 的空间正则算子 `construct_D(H, W)` 假定线性索引 = `col*H + row`
    （对应官方 MATLAB `reshape`，见 `osmagc/core/operators.py:37`）。
    若改用 C 序展平，空间邻接关系会和 OSMAGC 对不上，
    导致它的 OSAR 项算的是错的邻居 —— 而且不会报错，只是静默掉点。

口径（全部像素 vs 有标注像素）
------------------------------
- **聚类**：全体 N = H*W 个像素都参与（含背景），与 OSMAGC 官方一致。
- **评测**：只在 `gt > 0` 的像素上算指标（`labeled_mask()`）。
两者必须分开，否则数字和 OSMAGC 论文对不上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.io as sio

# 本包是**独立可携带**的：数据就放在同级目录 data/ 下，
# 不依赖任何外部项目（原平台指向 D:\cc\OSMAGC 等，已改为读包内）。
DATA = Path(__file__).resolve().parents[1] / "data"


# --------------------------------------------------------------------------- #
# 数据集规格
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ViewSpec:
    path: Path
    key: str | None = None          # None 表示取 .mat 里第一个非 header 变量
    name: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    views: tuple[ViewSpec, ...]
    gt: ViewSpec
    n_classes: int
    note: str = ""
    # 可选：含各视图「可用像素索引」的 .mat（变量名 `index`，MATLAB 1-based 的 cell）。
    # 缺失型数据集必须提供，否则缺失像素会被当成有效数据。
    availability: ViewSpec | None = None


def _v(rel: str, key: str | None, name: str) -> ViewSpec:
    return ViewSpec(path=DATA / rel, key=key, name=name)


SPECS: dict[str, DatasetSpec] = {
    # ---- Augsburg：唯一被 OSMAGC 与 FedRSMVC 都真跑过的数据集 ----
    "Augsburg": DatasetSpec(
        name="Augsburg",
        views=(
            _v("Augsburg/augsburg_hsi.mat", "augsburg_hsi", "HS"),
            _v("Augsburg/augsburg_sar.mat", "augsburg_sar", "SAR"),
            _v("Augsburg/augsburg_dsm.mat", "augsburg_dsm", "DSM"),
        ),
        gt=_v("Augsburg/augsburg_gt.mat", "augsburg_gt", "gt"),
        n_classes=7,
        note="332x485, 3视图 HS(180)+SAR(4)+DSM(1)，三个视图来自**不同传感器**，"
             "是真正的跨模态联邦场景。⚠️ DSM 为近似重建（非官方发布）。",
    ),
    # ---- Berlin：真·跨模态（HS + SAR 两个独立传感器），规模最大 ----
    "Berlin": DatasetSpec(
        name="Berlin",
        views=(
            _v("Berlin/berlin_hsi.mat", "berlin_hsi", "HS"),
            _v("Berlin/berlin_sar.mat", "berlin_sar", "SAR"),
        ),
        gt=_v("Berlin/berlin_gt.mat", "berlin_gt", "gt"),
        n_classes=8,
        note="1723x476 = 820,148 像素，2视图 HS(244)+SAR(4)。"
             "两个视图来自**不同传感器**，是真正的跨模态联邦场景。"
             "规模是其余数据集的 5~8 倍，跑起来慢、内存占用大。",
    ),
    # ---- Trento：FedRSMVC 主集 ----
    "Trento": DatasetSpec(
        name="Trento",
        views=(
            _v("Trento/Trento.mat", "HSI", "HS"),
            _v("Trento/Trento_lidar.mat", "Lidar", "LiDAR"),
        ),
        gt=_v("Trento/Trento_gt.mat", "GT", "gt"),
        n_classes=6,
        note="166x600, 2视图 HS(63)+LiDAR(2)。两个视图来自不同传感器。",
    ),
}


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class RSDataset:
    name: str
    view_names: list[str]
    cubes: list[np.ndarray]              # 每个 (H, W, B)
    gt: np.ndarray                       # (H, W) int32，0=背景
    n_classes: int
    note: str = ""
    meta: dict = field(default_factory=dict)
    # 各视图「可用像素」的 0-based 索引；None 表示完整数据（全部像素可用）。
    index: list[np.ndarray] | None = None

    # ---------------- 基础属性 ----------------
    @property
    def H(self) -> int:
        return int(self.gt.shape[0])

    @property
    def W(self) -> int:
        return int(self.gt.shape[1])

    @property
    def n_pixels(self) -> int:
        return self.H * self.W

    @property
    def n_views(self) -> int:
        return len(self.cubes)

    def view_dims(self) -> list[int]:
        return [int(c.shape[2]) for c in self.cubes]

    # ---------------- 展平 ----------------
    def flatten_view(self, v: int, order: str = "F") -> np.ndarray:
        """把第 v 个视图展平成 (N, B)。

        order='F' 是默认值，与 OSMAGC 的 `construct_D(H,W)` 列优先约定一致。
        """
        c = self.cubes[v]
        H, W, B = c.shape
        return c.reshape(H * W, B, order=order).astype(np.float32, copy=False)

    def flatten_gt(self, order: str = "F") -> np.ndarray:
        """展平 GT 成 (N,)，与 `flatten_view` 同序。"""
        return self.gt.reshape(-1, order=order).astype(np.int32)

    def coords(self, order: str = "F") -> np.ndarray:
        """返回 (N, 2) 的 (row, col)，与展平顺序对应。"""
        rr, cc = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        return np.stack(
            [rr.reshape(-1, order=order), cc.reshape(-1, order=order)], axis=1
        ).astype(np.int32)

    # ---------------- 掩码 ----------------
    def labeled_mask(self, order: str = "F") -> np.ndarray:
        """参与评测的像素（gt > 0），形状 (N,) bool。"""
        return self.flatten_gt(order) > 0

    def labeled_mask_2d(self) -> np.ndarray:
        return self.gt > 0

    # ---------------- 图结构信息 ----------------
    def grid_shape(self) -> tuple[int, int]:
        """供 `construct_D(H, W)` 使用。"""
        return self.H, self.W

    # ---------------- 缺失（视图内像素级） ----------------
    @property
    def is_complete(self) -> bool:
        """是否所有视图的所有像素都可用。"""
        return self.index is None

    def availability_2d(self, v: int) -> np.ndarray:
        """第 v 个视图的可用掩码，形状 (H, W)。完整数据时全 True。"""
        m = np.ones(self.n_pixels, dtype=bool)
        if self.index is not None:
            m[:] = False
            m[self.index[v]] = True
        return m.reshape(self.H, self.W, order="F")

    def availability_matrix(self) -> np.ndarray:
        """(N, V) bool 可用掩码，对应 OSMAGC 的 `Q`。"""
        return np.stack([self.availability_2d(v).reshape(-1, order="F")
                         for v in range(self.n_views)], axis=1)

    def missing_rate(self, v: int) -> float:
        return 1.0 - self.index[v].size / self.n_pixels if self.index is not None else 0.0

    def summary(self) -> str:
        dims = ", ".join(
            f"{n}({d})" for n, d in zip(self.view_names, self.view_dims())
        )
        n_lab = int(self.labeled_mask_2d().sum())
        miss = ""
        if self.index is not None:
            rates = "/".join(f"{100 * self.missing_rate(v):.0f}%" for v in range(self.n_views))
            miss = f" | 缺失率 {rates}"
        return (
            f"{self.name}: {self.H}x{self.W} = {self.n_pixels:,} 像素 | "
            f"{self.n_views} 视图 [{dims}] | {self.n_classes} 类 | "
            f"有标注 {n_lab:,} ({100 * n_lab / self.n_pixels:.1f}%){miss}"
        )


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #
def _read_mat(spec: ViewSpec) -> np.ndarray:
    if not spec.path.exists():
        raise FileNotFoundError(f"找不到数据文件：{spec.path}")
    d = sio.loadmat(spec.path)
    keys = [k for k in d if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{spec.path} 里没有可用的变量")
    key = spec.key if (spec.key and spec.key in d) else keys[0]
    arr = np.asarray(d[key])
    # 2D 视图（如 DSM）补一个通道维，统一成 (H, W, B)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3:
        raise ValueError(f"{spec.path}::{key} 形状异常：{arr.shape}")
    return arr


def load_dataset(name: str, dtype=np.float32) -> RSDataset:
    """按名字加载数据集，返回 `RSDataset`。

    形状不一致（H/W 对不上）会直接报错，避免静默错位。
    """
    if name not in SPECS:
        raise KeyError(f"未知数据集 {name!r}；可用：{sorted(SPECS)}")
    spec = SPECS[name]

    cubes = []
    for vs in spec.views:
        c = _read_mat(vs)
        cubes.append(np.ascontiguousarray(c.astype(dtype, copy=False)))

    gt_raw = _read_mat(spec.gt)
    gt = np.squeeze(gt_raw).astype(np.int32)
    if gt.ndim != 2:
        raise ValueError(f"{spec.gt.path} 的真值不是 2D：{gt.shape}")

    H, W = gt.shape
    for vs, c in zip(spec.views, cubes):
        if c.shape[:2] != (H, W):
            raise ValueError(
                f"[{name}] 视图 {vs.name} 尺寸 {c.shape[:2]} 与真值 {(H, W)} 不一致"
            )

    index = None
    if spec.availability is not None:
        index = _load_availability(spec.availability, n_views=len(cubes), N=H * W)

    return RSDataset(
        name=spec.name,
        view_names=[vs.name for vs in spec.views],
        cubes=cubes,
        gt=gt,
        n_classes=spec.n_classes,
        note=spec.note,
        index=index,
    )


def _load_availability(spec: ViewSpec, n_views: int, N: int) -> list[np.ndarray]:
    """加载各视图的可用像素索引，转成 0-based。

    用于**有像素级缺失**的数据集：mask 文件里 `index` 是 (1, V) 的 MATLAB cell，
    每个元素是该视图**可用**像素的 1-based 列优先索引。

    ⚠️ 本包当前三个数据集（Augsburg / Trento / Berlin）都是**完整**的，
    没有缺失，所以这段逻辑暂时用不上——保留它是为了以后接入带缺失的数据集
    （例如 MUUFL 的官方 0.5 缺失版本）时不用重写。
    """
    if not spec.path.exists():
        raise FileNotFoundError(f"找不到缺失掩码文件：{spec.path}")
    d = sio.loadmat(spec.path)
    if spec.key not in d:
        raise KeyError(f"{spec.path} 里没有变量 {spec.key!r}；可用：{[k for k in d if not k.startswith('__')]}")

    cell = d[spec.key]
    out: list[np.ndarray] = []
    # (1, V) object array 或 (V,) 均可
    flat = np.asarray(cell).ravel()
    if flat.size != n_views:
        raise ValueError(f"{spec.path}::{spec.key} 有 {flat.size} 个元素，期望 {n_views}")
    for i, item in enumerate(flat):
        arr = np.asarray(item).ravel().astype(np.int64)
        if arr.size and arr.min() >= 1:
            arr = arr - 1                     # 1-based -> 0-based
        if arr.size and (arr.min() < 0 or arr.max() >= N):
            raise ValueError(f"视图 {i} 的可用索引越界：[{arr.min()}, {arr.max()}] vs N={N}")
        arr = np.unique(arr)
        out.append(arr)
    return out


# --------------------------------------------------------------------------- #
# 预处理（各算法可自选，这里只提供通用选项）
# --------------------------------------------------------------------------- #
def normalize_view(X: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """对视图特征做逐列归一化。**返回新数组，不改动入参。**

    mode
    ----
    'zscore' : (x - mean) / std          —— 通用联邦 MVC 的常用做法
    'minmax' : (x - min) / (max - min)   —— 对应 MATLAB `mapminmax`
    'none'   : 原样返回副本

    ⚠️ 注意 `minmax` 在真实遥感数据上有坑：若某视图的取值范围不含 0
    （Augsburg 的 DSM 是 502~617），逐列 min-max 后再置零缺失像素，
    会把缺失标记混进有效值域。缺失场景下应先置零归一化。
    """
    X = np.asarray(X, dtype=np.float32)
    if mode == "none":
        return X.copy()
    if mode == "zscore":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        return (X - mu) / sd
    if mode == "minmax":
        lo = X.min(axis=0, keepdims=True)
        hi = X.max(axis=0, keepdims=True)
        rng = hi - lo
        rng[rng == 0] = 1.0
        return (X - lo) / rng
    raise ValueError(f"未知归一化方式 {mode!r}")


def load_all_views(ds: RSDataset, mode: str = "zscore") -> list[np.ndarray]:
    """加载全部视图并各自归一化，返回 [(N, B_v)] 列表。"""
    return [normalize_view(ds.flatten_view(v), mode) for v in range(ds.n_views)]
