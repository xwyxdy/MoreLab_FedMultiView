"""统一遥感联邦多视图聚类测试平台的公共模块。

⚠️ 这里在**包导入时**就把 BLAS/OpenMP 线程数钉死。任何脚本只要 `from common...`
导入本包，线程数即被固定，不依赖它是否先 import 了 numpy。
理由见 `common/threads.py` 顶部——线程数不固定会让同一个实验差出 5.9 个 ACC 点。
"""

from .threads import THREADS, apply as _pin_threads

_pin_threads()

__all__ = ["THREADS"]
