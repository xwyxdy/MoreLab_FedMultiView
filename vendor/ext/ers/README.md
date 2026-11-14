# ERS 超像素动态库

`ers.dll` 是 OSMAGC 复现所用 ERS（熵率超像素）分割的编译产物。

- **来源**：`D:\cc\OSMAGC\ext\ers\ers.dll`，由 `ers_wrapper.cpp`（作者仓库自带 C++ 源码，仅 `mex_ers.cpp` 引用 MATLAB API）用 MinGW-w64 g++ 编成独立 DLL。
- **限制**：Windows x64；需 64 位 Python。32 位 Python 加载会报 `OSError [WinError 193]`。
- **加载方式**：`vendor/osmagc/core/ers.py` 通过 `ctypes.CDLL` 调用，路径由 `Path(__file__).parents[2] / "ext" / "ers" / "ers.dll"` 解析到本文件。
- **许可证**：见仓库根 `vendor/LICENSE-OSMAGC`（MIT）。

若此文件缺失，OSMAGC 方法将无法运行，需从原 OSMAGC 项目复制 `ext/ers/ers.dll`。
