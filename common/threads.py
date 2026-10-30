"""固定 BLAS / OpenMP 线程数 —— 这是结果可复现的**前提条件**，不是性能优化。

为什么必须有这个模块
--------------------
实测：同一份数据、同一个配置、同一个随机种子，
只改 BLAS 线程数，得到三个不同的结果：

    | 线程数 | ACC    | NMI    | ARI    | Purity |
    |--------|--------|--------|--------|--------|
    | 24     | 0.6483 | 0.6862 | 0.5083 | 0.6567 |
    | 4      | 0.6014 | 0.6760 | 0.4789 | 0.6522 |
    | 1      | 0.6605 | 0.6934 | 0.5163 | 0.6688 |

**ACC 跨度 5.9 个点**；而常见调参网格里 top 端的 ACC 跨度通常只有 4~5 个点。
也就是说：不固定线程数，超参网格选出的"最优"可能只是线程调度的产物。

机理：BLAS 按线程划分改变浮点求和顺序 → 迭代轨迹分叉 → 最终聚类不同。
这是**系统性偏差，不是随机噪声**——重跑不会自我平均掉。

为什么用 threadpoolctl 而不是只设环境变量
----------------------------------------
`OPENBLAS_NUM_THREADS` 只在库**首次加载前**生效，而各脚本都是先 `import numpy`
再 `from common... import`，等本模块跑起来时 OpenBLAS 早就加载完了。
`threadpoolctl.threadpool_limits()` 可以在**运行时**改线程数，因此放在
`common/__init__.py` 里调用就能覆盖所有入口脚本，不依赖任何 import 顺序。

⚠️ 本模块**强制**覆盖各库的线程数（不用 `setdefault`）。允许外部环境变量随意
穿透会让同一批实验的不同记录落在不同协议下，正是本模块要防的事。
要换线程数请改 `RS_MVC_THREADS`，它会记进协议指纹。
"""

from __future__ import annotations

import os

# 线程数取值依据：
# - 1 最保守但实测慢 ~1.32x（332s vs 251s）
# - 4 在实测里与 24 线程同速（甚至更快），且任何 ≥4 核的机器都能复现
# 因此取 4。要改就用 RS_MVC_THREADS 环境变量，别在这里改。
THREADS: int = int(os.environ.get("RS_MVC_THREADS", "4"))

_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_applied = False


def apply(force: bool = False) -> int:
    """把所有 BLAS/OpenMP 线程数钉到 `THREADS`，返回实际生效值。

    幂等：重复调用只有第一次真正干活（除非 `force=True`）。
    """
    global _applied
    if _applied and not force:
        return THREADS

    for var in _ENV_VARS:
        os.environ[var] = str(THREADS)

    # 已经加载的库（numpy 的 OpenBLAS、sklearn 的 OpenMP）靠 threadpoolctl 改
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=THREADS)
    except ImportError:                                        # pragma: no cover
        # 没有 threadpoolctl 时环境变量对**尚未加载**的库仍然有效，
        # 但已经加载的改不了——这种情况下结果不可复现，必须显式警告。
        import warnings

        warnings.warn(
            "threadpoolctl 不可用，已加载的 BLAS 线程数无法固定，"
            "本次结果不可复现。请 `pip install threadpoolctl`。",
            RuntimeWarning,
            stacklevel=2,
        )
    _applied = True
    return THREADS


def describe() -> dict:
    """写进协议指纹用。"""
    return {"blas_threads": THREADS, "threads_env": os.environ.get("RS_MVC_THREADS", "(默认)")}
