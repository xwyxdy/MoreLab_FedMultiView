"""统一的实验协议与结果记录。

设计原则
--------
1. **一条记录 = 一次完整实验**（数据集 × 方法 × 划分 × 种子）。
2. 结果一律**追加**到 JSONL，可断点续跑、可去重（`already_done`）。
3. 记录里必须写清「协议指纹」：数据集、划分方式、客户端数、评测掩码口径。
   —— 没有这些，半年后没人知道表里那个数字是怎么来的。
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .threads import THREADS

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LOG = RESULTS_DIR / "runs.jsonl"


# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    # --- 协议指纹 ---
    dataset: str
    method: str
    partition: str                 # 'view' / '2x2' / 'stripe4V' / 'random4'
    partition_mode: str            # 'view' / 'region' / 'random'
    n_clients: int
    eval_mask: str = "labeled"     # 'labeled'（gt>0）或 'all'
    norm: str = "zscore"
    seed: int = 0
    # BLAS/OpenMP 线程数。实测同配置换线程数能差 5.9 个 ACC 点，所以它和种子
    # 一样属于"协议指纹"——不记下来，半年后没人知道表里那个数字是怎么来的。
    # 全局由 common/threads.py 钉死，这里只做记录。
    blas_threads: int = -1

    # --- 结果 ---
    ACC: float = float("nan")
    NMI: float = float("nan")
    ARI: float = float("nan")
    Purity: float = float("nan")

    # --- 元信息 ---
    status: str = "ok"             # 'ok' / 'error' / 'skip'
    note: str = ""
    runtime_s: float = float("nan")
    n_clusters: int = -1
    extra: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.blas_threads < 0:
            self.blas_threads = THREADS

    def key(self) -> tuple:
        """去重键：同一条协议只跑一次。"""
        return (self.dataset, self.method, self.partition, self.norm, self.seed)

    def line(self) -> str:
        if self.status != "ok":
            return f"[{self.dataset}/{self.partition}] {self.method:<22} {self.status}: {self.note}"
        return (
            f"[{self.dataset}/{self.partition}] {self.method:<22} "
            f"ACC={self.ACC:.4f}  NMI={self.NMI:.4f}  ARI={self.ARI:.4f}  PUR={self.Purity:.4f}"
            f"  ({self.runtime_s:.1f}s)"
        )


# --------------------------------------------------------------------------- #
def append_result(res: RunResult, path: Path = DEFAULT_LOG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")


def load_results(path: Path = DEFAULT_LOG) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def done_keys(path: Path = DEFAULT_LOG) -> set[tuple]:
    return {
        (r["dataset"], r["method"], r["partition"], r["norm"], r["seed"])
        for r in load_results(path)
        if r.get("status") == "ok"
    }


# --------------------------------------------------------------------------- #
def fingerprint() -> dict:
    """环境指纹，写进结果里方便复现排查。"""
    try:
        import sklearn
        sk = sklearn.__version__
    except Exception:                                  # noqa: BLE001
        sk = "?"
    try:
        import torch
        tch = torch.__version__
    except Exception:                                  # noqa: BLE001
        tch = "?"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sk,
        "torch": tch,
        "platform": platform.platform(),
    }


def fill_metrics(res: RunResult, y_true: np.ndarray, y_pred: np.ndarray) -> RunResult:
    """用统一指标模块填充四个指标（绝不使用各方法自带的指标代码）。"""
    from .metrics import evaluate

    m = evaluate(y_true, y_pred)
    res.ACC, res.NMI, res.ARI, res.Purity = m["ACC"], m["NMI"], m["ARI"], m["Purity"]
    return res
