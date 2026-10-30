"""环境自检：确认 BLAS 线程数已固定，且本机跑出来的数与他人一致。

为什么需要这个
--------------
本平台**强制固定 BLAS 线程数为 4**（见 common/threads.py）。原因是实测过：
同一份数据、同一个种子，只改线程数，ACC 能差 5.9 个点 ——

    24 线程 → ACC 0.6483
     4 线程 → ACC 0.6014
     1 线程 → ACC 0.6605

而超参网格里大多数配置之间的差距还不到 5 个点。也就是说：**线程数不固定，
测出来的差异里就混进了"环境差异"，分不清是谁造成的。**

线程数一改，所有历史结果都没法和新的比。所以拿到这个包第一件事就是跑本脚本。

用法
----
    python scripts/verify_threads.py

判读
----
  ① 线程数必须显示 4。不是 4 → 你没在包的根目录跑，或者环境变量被覆盖了。
  ② 打印出来的「校验和」记下来，和别人对一下。
     若线程数相同但校验和不同 → 对方的 CPU 指令集和你不同（如 AVX512 vs AVX2），
     这会造成末位差异，属正常，**不影响结论量级**。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPECTED_THREADS = 4


def main() -> int:
    print("=" * 78)
    print("环境自检")
    print("=" * 78)

    # ---- ① 本包规定的线程数 ----
    import common                                            # noqa: F401  触发线程固定
    from common.threads import THREADS

    print(f"\n[1] 本包固定线程数 : {THREADS}")
    print(f"    预期            : {EXPECTED_THREADS}")

    # ---- ② 运行时实际生效的线程数 ----
    try:
        from threadpoolctl import threadpool_info
        api = [(t.get("internal_api"), t.get("num_threads"))
               for t in threadpool_info() if t.get("num_threads")]
        print(f"\n[2] 实际生效的线程池 : {api}")
        ok2 = all(n == EXPECTED_THREADS for _, n in api)
        print(f"    {'✓ 一致' if ok2 else '✗ 不一致！BLAS 线程数没被固定住'}")
    except ImportError:
        print("\n[2] 未安装 threadpoolctl，无法读取实际线程数")
        print("    pip install threadpoolctl")

    # ---- ③ 环境变量 ----
    keys = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    print(f"\n[3] 相关环境变量 : "
          f"{ {k: os.environ.get(k, '(未设)') for k in keys} }")

    # ---- ④ 算一个可跨机器比对的校验和 ----
    # 固定种子的确定性矩阵乘法；线程数不同或 CPU 指令集不同，末位会有差异
    rng = np.random.default_rng(12345)
    A = rng.standard_normal((256, 256))
    B = rng.standard_normal((256, 256))
    C = A @ B
    checksum = float(np.sum(C))                       # 一个标量，方便口头比对
    print(f"\n[4] 矩阵乘法校验和 : {checksum:.6f}")
    print("    把你的这个数发给同伴对一下。")
    print("    - 完全一致 → 环境相同，结果应逐位一致")
    print("    - 末几位不同 → CPU 指令集不同，属正常，不影响结论")

    print("\n" + "=" * 78)
    if THREADS != EXPECTED_THREADS:
        print(f"⚠️ 线程数是 {THREADS}，但本包要求 {EXPECTED_THREADS}。")
        print("   检查 common/threads.py，或确认没有外部环境变量覆盖。")
        return 1
    print("✓ 线程数已固定为 4。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
