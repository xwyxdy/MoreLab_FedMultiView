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

    # 1. 取数据：F 序展平，与平台像素索引一一对应
    views = [np.asarray(ds.flatten_view(v, "F"), dtype=np.float32)
             for v in range(ds.n_views)]

    # 2. 拼接所有视图的特征
    X = np.concatenate(views, axis=1)

    # 3. k-means（固定种子——平台要求诚实读出协议，见 docs/协议与规范.md）
    labels = KMeans(n_clusters=ds.n_classes, n_init=10,
                    random_state=seed).fit_predict(X).astype(np.int32)

    return labels, {
        "runtime_s": time.time() - t0,
        "feature_dim": int(X.shape[1]),
        "备注": "示例算法，非对比方法",
    }
