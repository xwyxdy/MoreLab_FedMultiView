"""超像素预处理，对应官方 MATLAB 的 `funs/preData.m`。

对每个视图做：
    1. 立方体 (H,W,B) 按**列优先**展平成 (N, B)
    2. `mapminmax`  —— 把每一**行**（即每个像素的光谱向量）归一化到 [-1, 1]
    3. `pca(X)`    —— 对列中心化后做 SVD，取第一主成分方向 coeff(:,1)
    4. `Y_pca = X * coeff(:,1)`  —— ⚠️ 用的是**未中心化**的 X（见下）
    5. `im2uint8(mat2gray(...))` —— 归一化到 [0,1] 再转 uint8 灰度图
    6. `mex_ers`   —— 熵率超像素分割
    7. `labels + 1` —— 转成 1-based

⚠️ 三处易错点（都影响最终结果）：

1. **`mapminmax` 归一化的是「行」不是「列」**
   MATLAB 的 `mapminmax` 默认处理矩阵的每一**行**。X 是 (N, B) 形状，
   所以被归一化的是**每个像素的光谱向量**，不是每个波段。
   这一点反直觉（遥感领域更常见的做法是按波段归一化），但原码就是如此。

2. **`Y_pca = X*coeff(:,1)` 用的是未中心化的 X**
   `pca` 内部会中心化，但这里乘的是 mapminmax 之后的原始 X，属于常见写法。
   不能"顺手改成正确的主成分得分"——数值会变。

3. **第一主成分的符号不确定** ⭐
   SVD 的奇异向量符号取决于实现约定。符号一翻，`mat2gray` 后的灰度图就整体反相，
   ERS 分割结果随之不同，后面全变。本模块把符号做成参数 `pca_sign`，
   两种都跑，看哪个能复现论文数值（见 NOTES.md D7）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ers import mex_ers


@dataclass
class PreDataResult:
    """`preData` 的输出。"""

    sp_labels: list[np.ndarray]   # V 个 (H, W) int32，1-based 超像素标签
    pca_coeff: list[np.ndarray]   # V 个 (B, B)，第一主成分方向
    gray_img: list[np.ndarray]    # V 个 (H, W) uint8，喂给 ERS 的灰度图
    y_pca: list[np.ndarray]       # V 个 (N,)，第一主成分投影（未 mat2gray）


def mapminmax_masked(X: np.ndarray, avail: np.ndarray, ymin: float = -1.0,
                     ymax: float = 1.0) -> np.ndarray:
    """按**可用像素**统计每个波段 min/max 再做归一化（axis=0）。

    ⚠️ 为什么需要这个：`preData` 拿到的立方体里，缺失像素被置为 0。
    如果某个视图的真实取值范围**不含 0**（例如 Augsburg 的 DSM 是 502~617），
    那么这些 0 会把该波段的 min 拉到 0，把真实值压到一个极窄的区间里。
    实测 Augsburg 的 DSM 因此只剩 19% 的动态范围 —— 这个视图基本废掉。

    官方 MATLAB 代码用的是朴素 `mapminmax(X)`，同样会被污染；这里提供一个
    只用可用像素统计的版本，用来验证这是否就是 Augsburg 曲线形状不对的原因。
    """
    Xa = X[avail]
    xmin = Xa.min(axis=0, keepdims=True)
    xmax = Xa.max(axis=0, keepdims=True)
    rng = np.where(xmax - xmin == 0, 1.0, xmax - xmin)
    return (ymax - ymin) * (X - xmin) / rng + ymin


def mapminmax(X: np.ndarray, axis: int = 1, ymin: float = -1.0, ymax: float = 1.0) -> np.ndarray:
    """MATLAB `mapminmax` 的等价实现：沿 `axis` 归一化到 [ymin, ymax]。

    ⚠️ **方向存疑，做成开关**。
    MATLAB 的 `mapminmax` 文档写的是归一化每一**行**（`axis=1`，本函数默认）。
    但 `preData.m:13` 里 X 的形状是 (N, B)，按行归一化等于**把每个像素的光谱向量**
    归一化到 [-1,1]，这在遥感领域很反直觉（更常见的做法是按波段归一化，即 `axis=0`）。
    作者可能本意是后者。两种都实现，用最终指标判定（见 NOTES.md D13）。

    常数行/列（max == min）映射为 ymin —— 与 MATLAB 把除数置 1 的行为一致。
    """
    if axis == 1:
        xmin = X.min(axis=1, keepdims=True)
        xmax = X.max(axis=1, keepdims=True)
    else:
        xmin = X.min(axis=0, keepdims=True)
        xmax = X.max(axis=0, keepdims=True)
    rng = xmax - xmin
    rng = np.where(rng == 0, 1.0, rng)
    return (ymax - ymin) * (X - xmin) / rng + ymin


def mapminmax_rows(X: np.ndarray, ymin: float = -1.0, ymax: float = 1.0) -> np.ndarray:
    """按行归一化（MATLAB `mapminmax` 的字面行为）。"""
    return mapminmax(X, axis=1, ymin=ymin, ymax=ymax)


def pca_coeff(X: np.ndarray) -> np.ndarray:
    """MATLAB `pca(X)` 的等价实现（列中心化 + SVD），返回 coeff（B×B）。

    MATLAB 内部即 `svd(X - mean(X,1), 'econ')` 后取 V。
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # Xc 是 (N, B)，N >> B，full_matrices=False 得到 Vt 为 (B, B)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt.T


def mat2gray(A: np.ndarray) -> np.ndarray:
    """MATLAB `mat2gray`：按 min/max 线性归一化到 [0, 1]。"""
    lo, hi = float(A.min()), float(A.max())
    if hi == lo:
        return np.zeros_like(A, dtype=np.float64)
    return (A - lo) / (hi - lo)


def im2uint8(A: np.ndarray) -> np.ndarray:
    """MATLAB `im2uint8`：输入已在 [0,1] 的 double，输出 uint8(round(255*x))。"""
    return np.clip(np.round(np.asarray(A, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)


def _kmeans_labels(img: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    """消融 V1（w/o SAGI）用：把第一主成分图用 k-means 聚成 n_clusters 个区域。

    与 ERS 超像素**唯一的区别**是去掉了"空间感知"——
    k-means 只看灰度值、不管空间连续性，因此区域会碎裂。
    这样能在其余流程完全不变的前提下，单独隔离出 SAGI 的贡献。
    """
    from sklearn.cluster import KMeans

    H, W = img.shape
    # 列优先展平/还原，与代码库其余部分的像素索引约定保持一致
    feats = img.reshape(-1, 1, order="F").astype(np.float64)
    # ⚠️ 这里刻意不用 sklearn 的默认参数（n_init=10, max_iter=300）。
    # Berlin 有 820,148 个像素、要聚 100 类，用默认值单次要跑 20 分钟以上，
    # 光 V1 一个变体就占掉一小时。锚点区域只要大致合理即可（后续 initA 会在
    # 区域内取均值），所以降到 n_init=3 / max_iter=100，快一个数量级、划分基本不变。
    km = KMeans(n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
    labels = km.fit_predict(feats).astype(np.int32)
    return labels.reshape((H, W), order="F")


def predata(
    cubes: list[np.ndarray],
    num_pixel: list[int],
    pca_sign: int = 1,
    mapminmax_axis: int = 1,
    sagi_mode: str = "ers",
    seed: int = 0,
    avail_masks: list[np.ndarray] | None = None,
) -> PreDataResult:
    """对应 `funs/preData.m`。

    Parameters
    ----------
    cubes : list[np.ndarray]
        V 个 (H, W, Bv) 立方体。
    num_pixel : list[int]
        V 个超像素数量（对应 `osmagc.m` 的 `temp_num_Pixel` 一行）。
    pca_sign : int
        第一主成分的符号，+1 或 -1。
        ⚠️ 实测**对结果无影响**：ERS 的边权是 `|I_u - I_v|`，图像整体反相后
        所有边权不变，分割结果完全相同。保留此参数仅供核对。
    mapminmax_axis : int
        `mapminmax` 的归一化方向。1=按行（MATLAB 字面行为），0=按列（按波段，
        更符合遥感惯例，可能是作者本意）。见 NOTES.md D13。
    sagi_mode : str
        `'ers'`（默认，论文做法）：用熵率超像素做空间感知锚点。
        `'kmeans'`：**消融 V1（w/o SAGI）** —— 用 k-means 在同样的第一主成分图上
        聚类出 Mv 个区域代替超像素，去掉"空间感知"这一性质，其余流程不变。
    seed : int
        仅 `sagi_mode='kmeans'` 时用到的随机种子。

    Returns
    -------
    PreDataResult
    """
    V = len(cubes)
    if len(num_pixel) != V:
        raise ValueError(f"num_pixel 长度 {len(num_pixel)} != 视图数 {V}")
    if pca_sign not in (1, -1):
        raise ValueError(f"pca_sign 必须是 +1 或 -1，实际 {pca_sign}")

    sp_labels, coeffs, grays, y_pcas = [], [], [], []

    for v in range(V):
        data = np.asarray(cubes[v], dtype=np.float64)
        if data.ndim != 3:
            raise ValueError(f"视图 {v} 的立方体必须是 3D，实际 {data.ndim}D")
        n_row, n_col, dim = data.shape

        # 1. 列优先展平 (N, B)  —— 对应 reshape(data, nRow*nCol, dim)
        X = data.reshape(n_row * n_col, dim, order="F")

        # 2. mapminmax 归一化到 [-1, 1]（方向见参数说明）
        if avail_masks is not None:
            X = mapminmax_masked(X, avail_masks[v])
        else:
            X = mapminmax(X, axis=mapminmax_axis)

        # 3-4. 第一主成分方向；乘的是**未中心化**的 X
        coeff = pca_coeff(X)
        y_pca = X @ (pca_sign * coeff[:, 0])

        # 5. mat2gray -> im2uint8，reshape 回 (H, W)
        img = im2uint8(mat2gray(y_pca.reshape(n_row, n_col, order="F")))

        # 6. 分割成 Mv 个区域；7. 转 1-based（对应 preData.m:22 的 labels+1）
        if sagi_mode == "ers":
            labels = mex_ers(img, int(num_pixel[v])) + 1
        elif sagi_mode == "kmeans":
            labels = _kmeans_labels(img, int(num_pixel[v]), seed) + 1
        else:
            raise ValueError(f"sagi_mode 必须是 'ers' 或 'kmeans'，实际 {sagi_mode!r}")

        sp_labels.append(labels.astype(np.int32))
        coeffs.append(coeff)
        grays.append(img)
        y_pcas.append(y_pca)

    return PreDataResult(
        sp_labels=sp_labels, pca_coeff=coeffs, gray_img=grays, y_pca=y_pcas
    )
