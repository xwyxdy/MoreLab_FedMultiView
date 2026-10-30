# 遥感多视图聚类 · 统一测试平台

把各人的多视图聚类算法放到**同一批遥感数据**上、用**同一套指标**比较。
独立可运行，不依赖任何外部项目。

## 一次性准备（只做一次）

```bash
pip install -r requirements.txt     # 装 4 个依赖库：numpy / scipy / scikit-learn / threadpoolctl
python scripts/selfcheck.py         # 整包自检：语法 / 依赖 / 线程数 / 数据完整性 / 能否加载
```

自检全过就说明环境没问题。换电脑或重装环境才需要再做一遍。

> `selfcheck.py` 会逐项检查并给出 ✓/✗。
> 若只想确认线程数一条，用 `python scripts/verify_threads.py`。

## 日常使用

```bash
python scripts/run_all.py                                  # 不加参数 = 三个数据集全跑
python scripts/run_all.py --datasets Augsburg Berlin       # 只跑指定的几个
python scripts/run_all.py --datasets Trento                # 只跑一个
python scripts/run_all.py --methods my_algo                # 只跑指定算法（默认跑全部发现到的）
python scripts/run_all.py --seeds 0 1 2                    # 多种子（默认只跑 seed=0）
python scripts/run_all.py --list                           # 只看发现了哪些算法，不跑
cat results/runs.jsonl                                     # 看结果
```

默认三个数据集 = `Augsburg Trento Berlin`。

结果**跑完一组就立刻追加**到 `results/runs.jsonl`，中途被打断不丢已完成的；重跑会自动跳过已完成的。

## 加自己的算法

在 `methods/` 下建 `<算法名>.py`，写一个函数：

```python
def run_my_algo(ds, seed=0, **kwargs):
    views = [ds.flatten_view(v, "F") for v in range(ds.n_views)]   # F 序，与像素一一对应
    labels = my_method(views, n_clusters=ds.n_classes, seed=seed)  # 你的实现在这里
    return labels, {"runtime_s": ...}                              # labels: (N,) int32
```

然后 `python scripts/run_all.py --methods my_algo --datasets Trento`

**「每像素一个标签」这个约定，是在你自己的函数里落实的**——
公共层只提供数据和评分，不管方法内部怎么写；`labels` 由你保证返回 `(N,)`。
跑批时 `run_all.py` 会检查标签数是否等于 `ds.n_pixels`（只是检查，不是替你实现）。

细节照 `methods/_template.py` 抄。三条必须守的：

1. 不许用真值做任何选择
2. 不许在最后一步做后处理平滑
3. 像素顺序用 **F 序**；若你的代码是 C 序，必须 `labels_c.reshape(H, W).reshape(-1, order="F")` 转换

## 目录

```
common/     公共底座：数据 / 指标 / 划分 / 协议指纹 / 线程固定
methods/    算法，一人一个文件（_template.py 是模板，example_kmeans.py 是示例）
scripts/    run_all.py 跑批、selfcheck.py 整包自检、verify_threads.py 线程自检
data/       三个数据集（Augsburg / Trento / Berlin），已打包
results/    runs.jsonl 结果
docs/       协议与规范.md ★ 所有人必读
```

## 数据集

| 数据集 | 像素 | 视图 | 类别 | 视图来自 |
|---|---|---|---|---|
| Augsburg | 161,020 | HS(180)+SAR(4)+DSM(1) | 7 | 三个不同传感器 |
| Trento | 99,600 | HS(63)+LiDAR(2) | 6 | 两个不同传感器 |
| Berlin | 820,148 | HS(244)+SAR(4) | 8 | 两个不同传感器 |

**三个都是「不同传感器各持一个视图」——这才是联邦遥感的真实场景**（不同模态由不同机构/卫星持有）。

数据在 `data/` 下，`common/data.py` 直接读，不用改路径。

⚠️ **Berlin 比另外两个大 5~8 倍**（82 万像素），跑起来慢、内存占用大，调试时建议先用 Trento。

## 两个必须知道的

**线程数已固定为 4。** 实测只改线程数，ACC 能差 5.9 个点（超参差异还不到 5 点）。
改了所有历史结果都没法和新的比。用 `verify_threads.py` 能查出来。

**每条结果都带协议指纹**（数据集/种子/线程数）。比较数字前先确认指纹一致，否则不可比。

其余见 `docs/协议与规范.md`。
