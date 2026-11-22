import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import argparse
import importlib
import inspect
from pathlib import Path
from dataclasses import replace
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.data import load_dataset
from common.metrics import evaluate


def local_search(ds_name, algo_fn, param_pool, max_rounds=2):
    """
    通用的局部最优超参搜索器 (坐标下降法)
    """
    print(f"开始执行局部搜索 (Coordinate Descent) on {ds_name}...")
    ds = load_dataset(ds_name)
    gt = ds.flatten_gt("F")
    mask = gt > 0

    param_names = list(param_pool.keys())
    # 初始化当前参数为每个参数的第一个候选值
    current_params = {k: v[0] for k, v in param_pool.items()}
    history = []

    # 动态获取算法模块，用于获取默认参数
    module_name = algo_fn.__module__.split(".")[-1]
    mod = importlib.import_module(f"methods.{module_name}")

    def get_base_params():
        """辅助函数：获取算法的基础参数对象"""
        if hasattr(mod, "DEFAULT_PARAMS") and ds_name in mod.DEFAULT_PARAMS:
            return mod.DEFAULT_PARAMS[ds_name]
        elif hasattr(mod, "params_for"):
            return mod.params_for(ds)
        return None

    base_params = get_base_params()

    for round_idx in range(max_rounds):
        print(f"\n========== 第 {round_idx + 1} 轮坐标下降 ==========")
        changed = False

        for p_name in param_names:
            print(f"正在搜索参数 [{p_name}] ...")
            best_score = -float('inf')
            best_val = current_params[p_name]

            for candidate in param_pool[p_name]:
                test_params = current_params.copy()
                test_params[p_name] = candidate

                # 处理 JSON 列表转元组的问题（如 num_pixel）
                if isinstance(candidate, list):
                    test_params[p_name] = tuple(candidate)

                print(f"  测试 {test_params} ...", end="")

                # 核心修改：如果有基础参数对象，使用 replace 构造新对象；否则直接传参
                if base_params is not None:
                    new_params = replace(base_params, **test_params)
                    labels, _ = algo_fn(ds, seed=0, params=new_params)
                else:
                    labels, _ = algo_fn(ds, seed=0, **test_params)

                # 计算无监督评价指标：轮廓系数 (抽样 5000 个点加速)
                if len(np.unique(labels)) < 2:
                    score = -1.0  # 防止全聚成1类报错
                else:
                    sample_idx = np.random.choice(len(labels), min(5000, len(labels)), replace=False)
                    score = silhouette_score(labels[sample_idx].reshape(-1, 1), labels[sample_idx])

                print(f" 轮廓系数={score:.4f}")
                history.append((round_idx + 1, p_name, candidate, score))

                if score > best_score:
                    best_score = score
                    best_val = candidate

            # 更新最优参数
            if current_params[p_name] != best_val:
                current_params[p_name] = best_val
                changed = True
            print(f"--> [{p_name}] 更新为: {best_val}")

        if not changed:
            print(f"\n参数已稳定，提前结束搜索。")
            break

    # 最终使用真实标签评估最优参数的效果
    if base_params is not None:
        final_params = replace(base_params, **current_params)
        final_labels, _ = algo_fn(ds, seed=0, params=final_params)
    else:
        final_labels, _ = algo_fn(ds, seed=0, **current_params)

    final_acc = evaluate(gt[mask], final_labels[mask])["ACC"]

    print("\n=========================================")
    print(f"局部搜索结束！最优超参数组合: {current_params}")
    print(f"该参数下的最终真实 ACC: {final_acc:.4f}")
    print("=========================================")
    return history, current_params, final_acc  # 增加返回 final_acc


def plot_history(history, p_names, algo_name, ds_name, best_params, final_acc):
    """绘制搜索轨迹，并将图保存到 report 目录"""
    # 1. 确保输出目录存在（直接存到 results/report/）
    out_dir = ROOT / "results" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for p_name in p_names:
        p_history = [h for h in history if h[1] == p_name]
        scores = [h[3] for h in p_history]
        candidates = [h[2] for h in p_history]

        plt.plot(range(len(scores)), scores, marker='o', label=f'{p_name} search', linewidth=2)
        for i, c in enumerate(candidates):
            plt.annotate(f"{c}", (i, scores[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    plt.xlabel('Search Steps')
    plt.ylabel('Silhouette Score')
    plt.title(f'Local Search Trajectory: {algo_name} on {ds_name}')
    plt.grid(True, linestyle='--')
    plt.legend()

    # 2. 图片文件名加入数据集名，防止不同数据集覆盖
    fig_name = f"Local_Search_{algo_name}_{ds_name}.png"
    fig_path = out_dir / fig_name
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"搜索轨迹图已保存至 {fig_path}")
    plt.close()

    # 3. 将搜索记录写入 JSON
    record = {
        "algo": algo_name,
        "dataset": ds_name,
        "best_params": best_params,
        "final_acc": final_acc,
        "fig_name": fig_name
    }
    json_path = out_dir / "search_records.json"

    records = []
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                pass

    # 去重：如果同一个数据集同一个算法搜过了，就替换
    records = [r for r in records if not (r["algo"] == algo_name and r["dataset"] == ds_name)]
    records.append(record)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_search.json", help="JSON配置文件路径")
    ap.add_argument("--algo", default=None, help="如果指定，只搜这一个算法")
    args = ap.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        print(f"找不到 {config_path}，请先创建！")
        sys.exit(1)

    with open(config_path, "r") as f:
        configs = json.load(f)

    # 如果用户在命令行指定了 --algo，就过滤掉其他的
    if args.algo:
        configs = [c for c in configs if c["target_algorithm"] == args.algo]
        if not configs:
            print(f"在 JSON 中找不到算法: {args.algo}")
            sys.exit(1)

    # 循环执行队列中的每一个任务
    for task_idx, config in enumerate(configs):
        algo_name = config["target_algorithm"]
        ds_name = config["dataset"]
        param_pool = config["param_pool"]

        print(f"\n" + "=" * 50)
        print(f"开始执行任务 [{task_idx + 1}/{len(configs)}]: {algo_name} on {ds_name}")
        print("=" * 50)

        # 动态导入算法模块
        try:
            mod = importlib.import_module(f"methods.{algo_name}")
            algo_fn = getattr(mod, f"run_{algo_name}")
        except Exception as e:
            print(f"导入算法 {algo_name} 失败: {e}")
            continue

        # 执行局部搜索
        hist, best_p, final_acc = local_search(ds_name, algo_fn, param_pool)

        # 绘制该算法的搜索轨迹图并保存记录
        plot_history(hist, list(param_pool.keys()), algo_name, ds_name, best_p, final_acc)

    print("\n所有搜索任务执行完毕！")