"""数据加载与索引对齐，对应官方 MATLAB 的 `funs/findindex.m` / `funs/NormalizeData.m`。

官方数据布局（`official_code/data/`，以 MUUFL 为例）
---------------------------------------------------
    MUUFL_HSI_0.5.mat    -> HSI      (325, 220, 64)   多光谱立方体，缺失像素已置 0
    MUUFL_LiDAR_0.5.mat  -> LiDAR    (325, 220, 2)
    gt.mat               -> gt_muufl (325, 220)       含 0 号背景类
    MUUFL_0.5.mat        -> MissingStatus (N, V)
                            index         (1, V) cell，各视图可用像素的**1-based** 列优先索引
                            truelabel     (N, 1)

`osmagc.m` 先加载 3D 立方体作为 `data3D`（喂给 `preData` 做 PCA + ERS），
再加载 `data`/`index` 作为 `findindex` 的输入（构造 Q 与归一化后的 X）。

⚠️ 全部像素线性索引都是**列优先**的（与 MATLAB reshape 一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "official_code" / "data"


@dataclass
class Dataset:
    """一个多视图遥感数据集（含缺失）。"""

    name: str
    cubes: list[np.ndarray]      # V 个 (H, W, Bv) 立方体
    gt: np.ndarray               # (N,) int，含 0 号背景类，列优先展平
    index: list[np.ndarray]      # V 个数组，各视图**可用**像素的 0-based 索引
    rate: float
    H: int
    W: int
    V: int

    @property
    def N(self) -> int:
        return self.H * self.W

    @property
    def c(self) -> int:
        """类别数 = 唯一标签数 - 1（扣掉 0 号背景）。对应 `osmagc.m:33`。"""
        return len(np.unique(self.gt)) - 1

    def q_matrix(self) -> np.ndarray:
        """(N, V) uint8 可用掩码，对应 `findindex.m:14` 的 `ind`。"""
        return build_q(self.index, self.N, self.V)

    def flat_data(self) -> list[np.ndarray]:
        """各视图展平成 (Bv, N) 的原始数据，对应官方 `data` cell。"""
        return [
            cube.reshape((-1, cube.shape[2]), order="F").T.astype(np.float64)
            for cube in self.cubes
        ]


def build_q(index: list[np.ndarray], n: int, V: int) -> np.ndarray:
    """构造 (N, V) 的可用掩码。对应 `findindex.m:11-14`。

    MATLAB 里是 `ind(index{i}, i) = 1`（1-based），这里换成 0-based。
    """
    q = np.zeros((n, V), dtype=np.uint8)
    for v in range(V):
        q[index[v], v] = 1
    return q


def normalize_data(X: np.ndarray) -> np.ndarray:
    """对**每一列**做 L2 单位长度归一化。对应 `funs/NormalizeData.m:7-8`。

    ⚠️ 函数名叫 "NormalizeData" 但**不是** min-max 也不是 z-score，是 L2 归一化：
        ProcessData[:, i] = X[:, i] / max(1e-12, ||X[:, i]||)
    """
    norms = np.maximum(1e-12, np.linalg.norm(X, axis=0))
    return X / norms


def findindex(data: list[np.ndarray], index: list[np.ndarray]) -> tuple[list[np.ndarray], np.ndarray]:
    """对应 `funs/findindex.m`。

    Parameters
    ----------
    data : list[np.ndarray]
        V 个 (Bv, N) 数组（原始，含缺失处的 NaN/0）。
    index : list[np.ndarray]
        V 个可用像素的 0-based 索引。

    Returns
    -------
    X1 : list[np.ndarray]
        V 个 (Bv, N) 数组，每列已 L2 归一化；缺失列归一化后为全 0。
    Q : np.ndarray
        (N, V) uint8 可用掩码。
    """
    V = len(data)
    if len(index) != V:
        raise ValueError(f"data 有 {V} 个视图，index 有 {len(index)} 个")
    n = data[0].shape[1]

    X1: list[np.ndarray] = []
    for i in range(V):
        origin = np.asarray(data[i], dtype=np.float64)
        if origin.shape[1] != n:
            raise ValueError(f"视图 {i} 的像素数 {origin.shape[1]} != {n}")
        origin = np.where(np.isnan(origin), 0.0, origin)   # findindex.m:17
        X1.append(normalize_data(origin))

    return X1, build_q(index, n, V)


def load_official_muufl(
    rate: float = 0.5,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> Dataset:
    """加载官方仓库自带的 MUUFL 数据。

    目前官方只发布了 rate=0.5 的版本（`MUUFL_0.5.mat` 等）。其余缺失率可按
    `missing.make_missing_indices` 从完整立方体重新生成。

    Parameters
    ----------
    rate : float
        缺失率。当前只支持 0.5（官方发布的那一份）。
    data_dir : Path | str
        官方 data 目录。

    Returns
    -------
    Dataset
    """
    data_dir = Path(data_dir)
    tag = f"{rate:g}"
    hsi_path = data_dir / f"MUUFL_HSI_{tag}.mat"
    lidar_path = data_dir / f"MUUFL_LiDAR_{tag}.mat"
    meta_path = data_dir / f"MUUFL_{tag}.mat"
    gt_path = data_dir / "gt.mat"

    for p in (hsi_path, lidar_path, meta_path, gt_path):
        if not p.exists():
            raise FileNotFoundError(
                f"缺少官方数据文件：{p}\n"
                f"官方仓库只发布了 rate=0.5 的 MUUFL；其他缺失率需自行生成。"
            )

    hsi = np.asarray(sio.loadmat(hsi_path)["HSI"], dtype=np.float64)
    lidar = np.asarray(sio.loadmat(lidar_path)["LiDAR"], dtype=np.float64)
    gt_2d = np.asarray(sio.loadmat(gt_path)["gt_muufl"])

    meta = sio.loadmat(meta_path, squeeze_me=False)
    index_cell = meta["index"]
    # MATLAB 的 index 是 1-based，转成 0-based
    index = [
        np.asarray(index_cell[0, v]).ravel().astype(np.int64) - 1
        for v in range(index_cell.shape[1])
    ]

    H, W = gt_2d.shape
    gt = gt_2d.ravel(order="F").astype(np.int64)

    ds = Dataset(
        name="MUUFL",
        cubes=[hsi, lidar],
        gt=gt,
        index=index,
        rate=rate,
        H=H,
        W=W,
        V=2,
    )

    # 自检：index 指向的像素确实应在掩码内
    for v, idx in enumerate(ds.index):
        if idx.min() < 0 or idx.max() >= ds.N:
            raise ValueError(f"视图 {v} 的索引越界：[{idx.min()}, {idx.max()}]，N={ds.N}")

    return ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------- MUUFL 原始数据
# ⚠️ MUUFL 官方发布的原始场景是 **325×337×72**，而论文/官方代码用的是
# **325×220×64**（`construct_D(325, 220)`、`MUUFL_HSI_0.5.mat` 都是这个尺寸）。
# 官方仓库只发布了 0.5 缺失率那一份，其他缺失率必须从原始数据重建。
#
# 本模块用官方 0.5 数据里**未缺失的部分**反推出裁剪参数，并验证到 max|diff| = 0：
#     列偏移 = 0（取前 220 列）
#     波段   = raw 的第 4~67 波段（0-based，连续 64 个）
#     LiDAR  = Last Return 的 2 波段栅格，按 (325,337,2) **C 序** reshape
# 详见 NOTES.md F9。
MUUFL_RAW_SPEC = {
    "n_col": 220,        # 裁剪后的列数
    "band_start": 4,     # 起始波段（0-based）
    "n_band": 64,        # 保留波段数
    "lidar_return": 1,   # 0=First Return, 1=Last Return（实测是 Last）
    "lidar_order": "C",  # LiDAR z 向量的 reshape 顺序
}


def load_muufl_full(
    rate: float = 0.5,
    raw_path: Path | str | None = None,
    verify: bool = True,
) -> Dataset:
    """从 MUUFL 原始数据重建完整立方体，再按 `rate` 生成缺失。

    这样就能得到官方未发布的其它缺失率（0.1~0.9）。

    Parameters
    ----------
    rate : float
        缺失率。
    raw_path : Path | str | None
        原始 .mat 路径，默认 `data/MUUFL_raw/muufl_gulfport_campus_w_lidar_1.mat`。
        可从 https://github.com/GatorSense/MUUFLGulfport 下载。
    verify : bool
        是否用官方 `_0.5` 数据校验重建结果（强烈建议保持 True）。
    """
    raw_path = Path(raw_path) if raw_path else PROJECT_ROOT / "data/MUUFL_raw/muufl_gulfport_campus_w_lidar_1.mat"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"找不到 MUUFL 原始数据：{raw_path}\n"
            f"请从 https://github.com/GatorSense/MUUFLGulfport 下载 "
            f"MUUFLGulfportDataCollection/muufl_gulfport_campus_w_lidar_1.mat"
        )

    raw = sio.loadmat(raw_path, squeeze_me=False)["hsi"][0, 0]
    data = np.asarray(raw["Data"], dtype=np.float64)
    n_row, n_col_full, n_band_full = data.shape

    spec = MUUFL_RAW_SPEC
    cs, nb = spec["band_start"], spec["n_band"]
    hsi = data[:, : spec["n_col"], cs : cs + nb]

    lidar_cell = raw["Lidar"]
    st = lidar_cell[0, spec["lidar_return"]][0, 0]
    z = np.asarray(st["z"]).ravel().astype(np.float64)
    n_band_lidar = z.size // (n_row * n_col_full)
    lidar_full = z.reshape((n_row, n_col_full, n_band_lidar), order=spec["lidar_order"])
    lidar = lidar_full[:, : spec["n_col"], :]

    H, W = hsi.shape[:2]
    n = H * W
    V = 2
    from .missing import make_missing_indices

    index = make_missing_indices(n, V, rate)

    # 与官方一致：缺失像素在立方体里置 0（见 NOTES.md F5）
    cubes = []
    for v, cube in enumerate([hsi, lidar]):
        avail = np.zeros(n, dtype=bool)
        avail[index[v]] = True
        cubes.append(cube * avail.reshape((H, W), order="F")[:, :, None])

    gt_path = PROJECT_ROOT / "official_code/data/gt.mat"
    gt_2d = np.asarray(sio.loadmat(gt_path)["gt_muufl"])
    if gt_2d.shape != (H, W):
        raise ValueError(f"gt 形状 {gt_2d.shape} 与重建图像 {H}x{W} 不符")

    ds = Dataset(
        name="MUUFL", cubes=cubes, gt=gt_2d.ravel(order="F").astype(np.int64),
        index=index, rate=rate, H=H, W=W, V=V,
    )

    if verify and abs(rate - 0.5) < 1e-12:
        _verify_muufl_reconstruction(ds)

    return ds


def _verify_muufl_reconstruction(ds: Dataset) -> None:
    """用官方 `_0.5` 数据（未缺失部分）校验重建结果，不一致就直接报错。"""
    base = PROJECT_ROOT / "official_code/data"
    pairs = [
        (ds.cubes[0], base / "MUUFL_HSI_0.5.mat", "HSI"),
        (ds.cubes[1], base / "MUUFL_LiDAR_0.5.mat", "LiDAR"),
    ]
    for v, (cube, path, var) in enumerate(pairs):
        official = np.asarray(sio.loadmat(path)[var], dtype=np.float64)
        idx = ds.index[v]
        a = cube.reshape((-1, cube.shape[2]), order="F")[idx]
        b = official.reshape((-1, official.shape[2]), order="F")[idx]
        d = float(np.abs(a - b).max())
        if d > 0:
            raise AssertionError(
                f"MUUFL 重建校验失败：视图 {v} ({var}) 与官方 _0.5 数据不一致，max|diff| = {d}\n"
                f"裁剪参数可能不对，请检查 MUUFL_RAW_SPEC。"
            )

# 各数据集的文件清单（相对项目根目录）。Berlin / MDAS 待下载后补入。
DATASET_FILES: dict[str, dict] = {
    "MUUFL": {
        "dir": "official_code/data",
        "cubes": [("MUUFL_HSI_0.5.mat", "HSI"), ("MUUFL_LiDAR_0.5.mat", "LiDAR")],
        "gt": ("gt.mat", "gt_muufl"),
    },
    "Augsburg": {
        "dir": "data/Augsburg",
        "cubes": [
            ("augsburg_hsi.mat", "augsburg_hsi"),
            ("augsburg_sar.mat", "augsburg_sar"),
            ("augsburg_dsm.mat", "augsburg_dsm"),
        ],
        "gt": ("augsburg_gt.mat", "augsburg_gt"),
    },
    "Berlin": {
        # 来源：https://github.com/songyz2019/rs-fusion-datasets
        # 发布为 GitHub Release 资产（berlin-ouc.zip），比 TUM 原站/百度网盘好拿得多。
        # 实测完全匹配论文 Table 1：1723x476, HS(244), SAR(4), 8 类。
        "dir": "data/Berlin",
        "cubes": [("berlin_hsi.mat", "berlin_hsi"), ("berlin_sar.mat", "berlin_sar")],
        "gt": ("berlin_gt.mat", "berlin_gt"),
    },
    "MDAS": {
        "dir": "data/MDAS",
        "cubes": [
            ("mdas_hsi.mat", "mdas_hsi"),
            ("mdas_ms.mat", "mdas_ms"),
            ("mdas_sar.mat", "mdas_sar"),
            ("mdas_dsm.mat", "mdas_dsm"),
        ],
        "gt": ("mdas_gt.mat", "mdas_gt"),
    },
}


def load_dataset(name: str, rate: float = 0.5, root: Path | str | None = None) -> Dataset:
    """按名字加载数据集（统一入口）。

    MUUFL 走 `load_official_muufl`（官方自带 .mat 里已经存了缺失索引）；
    其余数据集从立方体构造，缺失索引按同一规则生成。

    Parameters
    ----------
    name : str
        数据集名，见 `DATASET_FILES`。
    rate : float
        缺失率。⚠️ 只有 MUUFL 官方发布了 0.5 这一份；其他缺失率需要**完整**
        （无缺失）的立方体才能生成。
    root : Path | str | None
        项目根目录，默认自动推断。
    """
    if name == "MUUFL":
        # 有原始数据就用它（能生成任意缺失率，且会与官方 0.5 自动校验）；
        # 否则退回官方发布的 0.5 那一份（仅支持 rate=0.5）。
        raw = (Path(root) if root else PROJECT_ROOT) / "data/MUUFL_raw/muufl_gulfport_campus_w_lidar_1.mat"
        if raw.exists():
            return load_muufl_full(rate, raw)
        if abs(rate - 0.5) > 1e-12:
            raise FileNotFoundError(
                f"MUUFL 在 rate={rate} 下需要原始数据 {raw}（官方只发布了 0.5）。\n"
                f"请从 https://github.com/GatorSense/MUUFLGulfport 下载。"
            )
        return load_official_muufl(rate, (root or PROJECT_ROOT) / "official_code" / "data")

    if name not in DATASET_FILES:
        raise KeyError(f"未知数据集 {name!r}，可选：{list(DATASET_FILES)}")

    spec = DATASET_FILES[name]
    base = (Path(root) if root else PROJECT_ROOT) / spec["dir"]
    paths, var_names = [], []
    for fn, var in spec["cubes"]:
        p = base / fn
        if not p.exists():
            raise FileNotFoundError(
                f"数据集 {name} 缺少文件：{p}\n"
                f"请先准备数据（见 README.md 的数据来源表）。"
            )
        paths.append(p)
        var_names.append(var)

    gt_fn, gt_var = spec["gt"]
    gt_path = base / gt_fn
    if not gt_path.exists():
        raise FileNotFoundError(f"数据集 {name} 缺少 gt 文件：{gt_path}")

    return load_cubes_from_mat(paths, var_names, gt_path, gt_var, name, rate)


def load_cubes_from_mat(
    paths: list[Path | str],
    var_names: list[str],
    gt_path: Path | str,
    gt_var: str,
    name: str,
    rate: float,
) -> Dataset:
    """从一组 (立方体 .mat, gt .mat) 构造 Dataset（用于 Berlin / Augsburg / MDAS）。

    缺失索引需另行生成（调用 `missing.make_missing_indices`）。
    """
    cubes = []
    for p, var in zip(paths, var_names):
        arr = np.asarray(sio.loadmat(p)[var], dtype=np.float64)
        if arr.ndim == 2:                     # (H, W) 的单波段视图
            arr = arr[:, :, None]
        cubes.append(arr)

    H, W = cubes[0].shape[:2]
    for i, c in enumerate(cubes):
        if c.shape[:2] != (H, W):
            raise ValueError(f"视图 {i} 的空间尺寸 {c.shape[:2]} 与 {H}x{W} 不一致")

    gt_2d = np.asarray(sio.loadmat(gt_path)[gt_var])
    if gt_2d.shape != (H, W):
        raise ValueError(f"gt 形状 {gt_2d.shape} 与图像 {H}x{W} 不一致")

    n = H * W
    V = len(cubes)
    from .missing import make_missing_indices

    index = make_missing_indices(n, V, rate)

    # ⚠️ 关键：把缺失像素在立方体里**置 0**。
    # 官方 osmagc.m 传给 preData 的 data3D 就是已掩码的立方体
    # （见 MUUFL_HSI_0.5.mat：缺失处全为 0），PCA 与 ERS 都跑在掩码后的图上。
    # 若不掩码，超像素分割相当于"偷看"了缺失区域，与论文设定不符。
    masked = []
    for v, cube in enumerate(cubes):
        avail = np.zeros(n, dtype=bool)
        avail[index[v]] = True
        avail_2d = avail.reshape((H, W), order="F")
        masked.append(cube * avail_2d[:, :, None])

    return Dataset(
        name=name,
        cubes=masked,
        gt=gt_2d.ravel(order="F").astype(np.int64),
        index=index,
        rate=rate,
        H=H,
        W=W,
        V=V,
    )
