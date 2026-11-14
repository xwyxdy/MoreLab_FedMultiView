# vendor/：第三方复现代码

本目录只放**论文方法的实现**，不放平台公共逻辑（公共逻辑在 `common/`）。

| 子目录 | 论文 | 官方代码 | 本目录内容 |
|---|---|---|---|
| `osmagc/` | OSMAGC（CVPR 2026，Zhang Yongshan 等） | GitHub `ZhangYongshan/OSMAGC`（MATLAB） | 逐行转写的 Python 复现（含 `FORMULAS.md`/`NOTES.md` 的差异记录） |
| `etap/` | ETAP（AAAI 2026，Zhang Yongshan 等） | GitHub `ZhangYongshan/ETAP`（MATLAB） | 按官方 `main.m` + `funs/*.m` 逐函数转写的 Python 实现 |
| `samvgc/` | SAMVGC（AAAI 2025，Guan Renxiang 等） | **无公开代码** | 按论文公式实现的 PyTorch 版本 |
| `ext/ers/` | ERS 超像素（Liu et al. 2011） | 随 OSMAGC 官方仓库提供 C++ 源码 | 编译好的 `ers.dll`（Windows x64） |

## 说明

- **OSMAGC / ETAP 的官方实现是 MATLAB**；本机无 MATLAB，`vendor/` 里是 Python 转写版，
  不是直接下载的官方代码。OSMAGC 的转写版已与官方数值对拍。
- **SAMVGC 没有官方代码**，`vendor/samvgc/` 是按论文公式重实现，绝对精度与论文有差距
  （论文未给全超参）。
- `ers.dll` 是二进制产物，由官方 C++ 源码编译；仅 Windows x64 / 64 位 Python 可加载。
- 许可证：`LICENSE-OSMAGC`（MIT）。ETAP 官方仓库未附带 LICENSE 文件；SAMVGC 无代码。

## 加新方法的约定

1. 方法实现放 `vendor/<名>/`，平台适配器放 `methods/<名>.py`。
2. `methods/<名>.py` 只做三件事：定义超参、转换数据、调用 `vendor` 并返回 `(labels, detail)`。
3. 不要在 `vendor/` 里 import `common/`，也不要写死包外绝对路径。
