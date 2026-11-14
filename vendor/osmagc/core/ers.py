"""ERS (Entropy Rate Superpixel) 超像素分割的 Python 接口。

对应官方 MATLAB 的 `mex_ers(double(img), nC)`（见 `funs/preData.m:21`）。

官方 mex 网关里有两条调用分支，本实现对应 **nrhs==2** 那条：
    lambda = 0.5, sigma = 5.0, kernel = 0   （硬编码，见 official_code/.../ERS/mex_ers.cpp:60-66）

Python 侧用 ctypes 调用独立编译的 `ext/ers/ers.dll`，而不是 Python C 扩展 ——
因为本机 Python 是 MSVC 构建、只有 MinGW g++，两者 ABI 不兼容。

索引约定（与官方网关逐字一致，勿改）：
    输入按列优先(Fortran order)展平 -> C 侧读入
    C 侧输出按列优先写回
    因此本模块对 (H,W) 数组用 order='F' 展平传入、order='F' reshape 取回
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_DLL_PATH = Path(__file__).resolve().parents[2] / "ext" / "ers" / "ers.dll"

_lib: ctypes.CDLL | None = None


class ErsNotBuiltError(RuntimeError):
    """ERS DLL 尚未编译。"""


def _load_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    if not _DLL_PATH.exists():
        raise ErsNotBuiltError(
            f"找不到 ERS 动态库：{_DLL_PATH}\n"
            f"请从原 OSMAGC 项目复制 ext/ers/ers.dll 到本路径。\n"
            f"（需要 64 位 Python / Windows x64）"
        )

    try:
        lib = ctypes.CDLL(str(_DLL_PATH))
    except OSError as exc:
        if getattr(exc, "winerror", None) == 193:
            raise ErsNotBuiltError(
                f"ERS 动态库加载失败（位数不符）：{_DLL_PATH}\n"
                f"WinError 193 通常表示 Python 是 32 位，需用 64 位 Python。"
            ) from exc
        raise ErsNotBuiltError(
            f"ERS 动态库加载失败：{_DLL_PATH}\n{exc}"
        ) from exc
    lib.ers_segment.argtypes = [
        ctypes.POINTER(ctypes.c_ubyte),  # img
        ctypes.c_int,                    # width
        ctypes.c_int,                    # height
        ctypes.c_int,                    # nC
        ctypes.POINTER(ctypes.c_int),    # out
    ]
    lib.ers_segment.restype = ctypes.c_int
    lib.ers_label_count.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.ers_label_count.restype = ctypes.c_int

    _lib = lib
    return lib


def mex_ers(img: np.ndarray, n_superpixels: int) -> np.ndarray:
    """对灰度图做熵率超像素分割。

    Parameters
    ----------
    img : np.ndarray
        2D 灰度图，取值 0..255（对应 MATLAB 的 uint8 图像）。
    n_superpixels : int
        期望的超像素个数（MATLAB 的 nC）。

    Returns
    -------
    np.ndarray
        (H, W) 的 int32 标签矩阵，标签从 **0** 开始。
        注意：`preData.m:22` 在调用后做了 `labels = labels + 1`，本函数**不做**这一步。

    Notes
    -----
    返回的标签个数可能略微多于或少于 n_superpixels —— ERS 是贪心合并算法，
    nC 是目标而非保证值。官方 MATLAB 实现也有同样行为。
    """
    if img.ndim != 2:
        raise ValueError(f"img 必须是 2D 灰度图，实际维度 {img.ndim}")

    height, width = img.shape
    if img.dtype != np.uint8:
        img = np.clip(np.round(img), 0, 255).astype(np.uint8)

    # 列优先展平 —— 对应 MATLAB 的列优先内存布局
    flat = np.ascontiguousarray(img.ravel(order="F"))
    assert flat.size == height * width

    out = np.zeros(height * width, dtype=np.int32)

    lib = _load_lib()
    rc = lib.ers_segment(
        flat.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
        ctypes.c_int(width),
        ctypes.c_int(height),
        ctypes.c_int(int(n_superpixels)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )
    if rc != 0:
        raise RuntimeError(f"ers_segment 失败，返回码 {rc}（-1 参数非法，-2 nC 超过像素数）")

    # 列优先 reshape 回 (H, W)
    return out.reshape((height, width), order="F")


def label_count(labels: np.ndarray) -> int:
    """返回标签矩阵里实际出现的超像素个数。"""
    flat = np.ascontiguousarray(labels.ravel(order="F"), dtype=np.int32)
    return int(lib_count(flat))


def lib_count(flat_int32: np.ndarray) -> int:
    lib = _load_lib()
    return lib.ers_label_count(
        flat_int32.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(flat_int32.size),
    )
