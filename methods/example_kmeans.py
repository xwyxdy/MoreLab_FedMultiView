"""【示例算法】把各视图拼起来直接做 k-means —— 这是「hello world」，不是对比方法。

⚠️ 这个文件的作用只有一个：**让包拿到手就能跑通**，验证环境没问题。
   它**不是**本平台要对比的任何算法，出报告时请把它的结果剔除，或标注为「示例」。

看它怎么写的，就能照着写你自己的算法（也见 methods/_template.py）。
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import KMeans


def run_example_kmeans(ds, seed: int = 0, **_):
    """拼接所有视图 → k-means → 每像素一个标签。

    这是最朴素的「多视图」做法（其实只是把特征拼起来），
    相当于一条基线，用来确认数据加载 / 指标计算 / 落盘链路是通的。
    """
    t0 = time.time()

    # ── 第 1 步：取数据 ────────────────────────────────────────────────
    # ds.cubes[v] 是 (H, W, B_v) 的立方体；flatten_view(v, "F") 展平成 (N, B_v)。
    # ⚠️ 必须是 F 序（列优先），这样第 i 行才对应平台上第 i 个像素。
    #    用 numpy 默认的 C 序会让标签整体错位，指标掉到随机水平。
    views = [np.asarray(ds.flatten_view(v, "F"), dtype=np.float32)
             for v in range(ds.n_views)]

    # ── 第 2 步：拼成一个特征矩阵（这就是最朴素的“多视图”，只拼不融合）──
    X = np.concatenate(views, axis=1)            # (N, ΣB_v)

    # ── 第 3 步：聚类，得到每像素一个标签 ──────────────────────────────
    # 诚实读出协议：固定 random_state、n_init=10，只用 inertia 选初始化，
    # 绝不用真值挑“ACC 最高的那一次”（见 docs/协议与规范.md 第三节）。
    labels = KMeans(n_clusters=ds.n_classes, n_init=10,
                    random_state=seed).fit_predict(X).astype(np.int32)

    return labels, {
        "runtime_s": time.time() - t0,
        "feature_dim": int(X.shape[1]),
        "备注": "示例算法，非对比方法",
    }
