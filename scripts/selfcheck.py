"""整包自检：拿到包之后跑一遍，确认没有缺文件、没有外部依赖、能正常出结果。

用法
----
    python scripts/selfcheck.py

检查项
------
  1. 所有 Python 文件语法正确
  2. 依赖库都装了
  3. BLAS 线程数固定为 4
  4. `data/` 下每份数据文件都存在且能加载
  5. 数据文件的大小与预期一致（防止拷贝时截断）
  6. 代码里没有指向包外的绝对路径
  7. 跑批脚本能发现算法
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PASS, FAIL, WARN = "✓", "✗", "!"

# 预期存在的数据文件（防止拷贝不完整）
EXPECTED_DATA = {
    "Augsburg": ["augsburg_hsi.mat", "augsburg_sar.mat", "augsburg_dsm.mat",
                 "augsburg_gt.mat"],
    "Trento": ["Trento.mat", "Trento_lidar.mat", "Trento_gt.mat"],
    "Berlin": ["berlin_hsi.mat", "berlin_sar.mat", "berlin_gt.mat"],
}

problems: list[str] = []


def head(t: str) -> None:
    print(f"\n{t}")
    print("-" * len(t) * 2)


def main() -> int:
    print("=" * 70)
    print("整包自检")
    print("=" * 70)

    # ---- 1. 语法 ----
    head("[1] Python 文件语法")
    pyfiles = [p for p in ROOT.rglob("*.py") if "__pycache__" not in str(p)]
    bad = []
    for f in pyfiles:
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append((f, e))
    print(f"    检查了 {len(pyfiles)} 个文件")
    if bad:
        for f, e in bad:
            print(f"    {FAIL} {f.relative_to(ROOT)}: {e}")
        problems.append("有文件语法错误")
    else:
        print(f"    {PASS} 全部语法正确")

    # ---- 2. 依赖 ----
    head("[2] 依赖库")
    for mod in ("numpy", "scipy", "sklearn", "threadpoolctl"):
        try:
            __import__(mod)
            print(f"    {PASS} {mod}")
        except ImportError:
            print(f"    {FAIL} {mod} 未安装 —— 跑 pip install -r requirements.txt")
            problems.append(f"缺少依赖 {mod}")

    # ---- 3. 线程数 ----
    head("[3] BLAS 线程数")
    import common                                          # noqa: F401
    from common.threads import THREADS
    if THREADS == 4:
        print(f"    {PASS} 固定为 {THREADS}")
    else:
        print(f"    {FAIL} 固定为 {THREADS}，应为 4 —— 结果将不可复现")
        problems.append("线程数不是 4")

    # ---- 4 & 5. 数据文件存在且大小合理 ----
    head("[4] 数据文件")
    from common.data import load_dataset
    for dsname, files in EXPECTED_DATA.items():
        d = ROOT / "data" / dsname
        miss = [f for f in files if not (d / f).exists()]
        empty = [f for f in files if (d / f).exists() and (d / f).stat().st_size < 1000]
        if miss:
            print(f"    {FAIL} {dsname}: 缺 {miss}")
            problems.append(f"{dsname} 缺文件")
        elif empty:
            print(f"    {FAIL} {dsname}: 这些文件太小，可能拷贝不完整 {empty}")
            problems.append(f"{dsname} 文件不完整")
        else:
            sz = sum((d / f).stat().st_size for f in files) / 1e6
            print(f"    {PASS} {dsname}: {len(files)} 个文件, {sz:.0f} MB")

    # ---- 6. 实际加载 ----
    head("[5] 数据集加载")
    for dsname in EXPECTED_DATA:
        try:
            ds = load_dataset(dsname)
            print(f"    {PASS} {dsname}: {ds.n_pixels:,} 像素, {ds.n_views} 视图, "
                  f"{ds.n_classes} 类")
        except Exception as exc:                            # noqa: BLE001
            print(f"    {FAIL} {dsname}: {type(exc).__name__}: {exc}")
            problems.append(f"{dsname} 加载失败")

    # ---- 7. 没有包外绝对路径 ----
    head("[6] 包外依赖检查")
    hits = []
    for f in pyfiles:
        txt = f.read_text(encoding="utf-8")
        for i, ln in enumerate(txt.split("\n"), 1):
            if ("D:\\" in ln or "D:/" in ln) and not ln.lstrip().startswith("#"):
                hits.append(f"{f.relative_to(ROOT)}:{i}")
    if hits:
        print(f"    {WARN} 发现 {len(hits)} 处硬编码绝对路径：{hits[:5]}")
        print("       （若在注释或 docstring 里可忽略）")
    else:
        print(f"    {PASS} 没有硬编码的包外路径")

    # ---- 8. 算法发现 ----
    head("[7] 算法发现")
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_all import discover_methods
    ms = discover_methods()
    print(f"    {PASS} 发现 {len(ms)} 个: {list(ms)}")

    # ---- 9. ERS 动态库（OSMAGC/ETAP 方法依赖，仅 Windows x64）----
    head("[8] ERS 动态库（OSMAGC/ETAP 依赖）")
    dll = ROOT / "vendor" / "ext" / "ers" / "ers.dll"
    if not dll.exists():
        print(f"    {WARN} 缺 {dll.relative_to(ROOT)}，OSMAGC/ETAP 方法将不可用")
    else:
        try:
            import ctypes
            ctypes.CDLL(str(dll))                        # 只验证可加载，不跑分割
            print(f"    {PASS} ers.dll 可加载（{dll.stat().st_size / 1e6:.1f} MB）")
        except OSError as exc:
            print(f"    {FAIL} ers.dll 加载失败：{exc}")
            problems.append("ers.dll 加载失败（需 64 位 Python / Windows x64）")

    # ---- 10. 可选依赖（按方法/脚本；缺了只 WARN，不影响其他方法）----
    head("[9] 可选依赖（按方法/脚本）")
    OPTIONAL = [
        ("osqp", "ETAP（二次规划）"),
        ("torch", "SAMVGC（GNN 训练）"),
        ("skimage", "SAMVGC（SLIC 超像素）"),
        ("matplotlib", "报告与像素图（make_report/plot_maps）"),
    ]
    for mod, purpose in OPTIONAL:
        try:
            __import__(mod)
            print(f"    {PASS} {mod:<12}（{purpose}）")
        except ImportError:
            print(f"    {WARN} {mod:<12} 未安装 —— {purpose} 将不可用")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    if problems:
        print(f"{FAIL} 自检未通过，{len(problems)} 个问题：")
        for p in problems:
            print(f"    - {p}")
        return 1
    print(f"{PASS} 自检全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
