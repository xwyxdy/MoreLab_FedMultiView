"""【算法适配器模板】—— 加一个新算法，照这个文件抄。

这个文件不是能跑的算法，是**写新算法时的骨架**。复制成 `<你的算法名>.py` 后填内容。

═══════════════════════════════════════════════════════════════════════════
 唯一必须遵守的约定：接口
═══════════════════════════════════════════════════════════════════════════

每个算法文件对外只暴露一个函数，签名固定：

    run_<算法名>(ds, seed=0, **kwargs) -> (labels, detail)

      ds      : RSDataset，平台统一的数据集对象（见 common/data.py）
      seed    : 随机种子
      labels  : (N,) int32    ★ 每像素一个簇标签，N = 图像总像素数
      detail  : dict          元信息（耗时、超参、通信量……随便放）

三条硬规矩（违反的话结果就和其他方法不可比了）：

  ① 不许用真值做任何选择
     典型反例：跑多次 k-means，用真值挑 ACC 最高的那一次。
     这是「考试时把答案摊在桌上挑抄得最像的」，属于作弊。

  ② 不许在最后一步做后处理平滑
     broadcast 到像素后不许再滤波——那已经是另一个方法了。

  ③ 内部随便怎么写，平台不管
     锚点图 / 超像素 / 对比学习 / 神经网络……这是算法本身的设计自由。

═══════════════════════════════════════════════════════════════════════════
 数据对象长什么样（RSDataset）
═══════════════════════════════════════════════════════════════════════════

    ds.name          'Augsburg' / 'Trento' / 'Berlin'
    ds.n_views       视图数 V
    ds.n_pixels      像素数 N
    ds.H, ds.W       栅格高宽
    ds.n_classes     类别数 K

    ds.cubes         视图列表，每个 (H, W, B_v) 的 float 数组
    ds.gt            (H, W) int32 真值，0 = 背景

    ds.flatten_view(v, order='F')   把第 v 个视图展平成 (N, B_v)
    ds.flatten_gt(order='F')        把真值展平成 (N,)

★ 顺序约定：平台统一用 **F 序（列优先）**。
  如果你用的第三方代码是 C 序（行优先，比如 numpy 默认的 reshape），
  必须做置换，否则标签整体错位、指标会掉到随机水平：

      labels_platform = labels_c.reshape(H, W).reshape(-1, order='F')
"""

from __future__ import annotations

import time

import numpy as np


def run_template(ds, seed: int = 0, verbose: bool = False):
    """跑一次算法。把下面的 `_your_method` 换成你自己的实现。"""
    t0 = time.time()

    # ---- 1. 取数据（F 序展平，索引与平台像素一一对应）----
    views = [np.asarray(ds.flatten_view(v, "F"), dtype=np.float32)
             for v in range(ds.n_views)]          # list of (N, B_v)
    gt = ds.flatten_gt("F")                        # (N,) 只用来最后评分，不许参与算法

    # ---- 2. 调你自己算法的子函数（想怎么调就怎么调）----
    labels = _your_method(views, n_clusters=ds.n_classes, seed=seed)

    # ---- 3. 收尾：形状与类型对齐 ----
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    assert labels.size == ds.n_pixels, \
        f"标签数 {labels.size} != 像素数 {ds.n_pixels}"

    detail = {
        "runtime_s": time.time() - t0,
        "备注": "这里写清楚你的算法对数据有什么硬性要求（比如需要完整栅格）",
    }
    return labels, detail


def _your_method(views, n_clusters, seed):
    """你自己的算法实现 —— 这里随便写。

    参考写法：
        from sklearn.cluster import KMeans
        X = np.concatenate(views, axis=1)
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)

    ⚠️ 注意：k-means 这类需要随机初始化的步骤，**必须固定 random_state**，
       否则每次结果都不一样（本平台的诚实读出协议要求 n_init=10、固定种子）。
    """
    raise NotImplementedError("把你的算法写在这里")


# ═══════════════════════════════════════════════════════════════════════════
#  如果你的算法是「联邦」的，再加一个函数（可选）
# ═══════════════════════════════════════════════════════════════════════════
#
# 联邦方法要额外说明两件事：客户端怎么分、每轮传什么。
#
#     def run_<算法名>_federated(ds, part, seed=0):
#         """
#         part : Partition 对象（见 common/partition.py）
#                part.shape  划分形状，如 'view' / '2x2'
#                part.mode   'view' / 'region' / 'random'
#                part.clients  客户端列表，每个含它持有的视图或样本
#
#         返回的 detail 里**必须**带上通信量：
#                {"comm_bytes": 总字节数, ...}
#         """
#
# 这样所有人的联邦方法都用同一套通信量口径，才比得了「谁的联邦化代价小」。
#
# ⚠️ 联邦化的正确性验证（强烈建议做）：
#     联邦版和集中版的中间结果应当**逐元素相同**（若算法本身可无损切分）。
#     参考：原平台里 OSMAGC 的联邦版就是这样验证的——10 轮迭代 max|ΔH| = 0。
#     如果做不到无损，就要在报告里说明「损失来自联邦化本身」。
