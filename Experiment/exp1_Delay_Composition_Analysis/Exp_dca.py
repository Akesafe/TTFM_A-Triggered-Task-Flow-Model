import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# --- 1. 环境设置与路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../Optimization_Methods', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from Experiment_Setup.Model_Generation.Generate_TTFM_Task_RANDOM import TaskDAG
    from Experiment_Setup.Model_Generation.Generate_TTFM_Resource import NetworkGenerator
    from TTFM_Simulation_main import TaskFlowSimulator
    from Optimization_Methods.TS_Algorithms import TabuSearch
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# Matplotlib 学术风格设置
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,  # 标题稍微调小一点适配窄图
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (6, 5)  # <--- 修改点：图变窄，比例更协调
})

# --- 2. 实验参数配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "composition_analysis_output")

# 实验场景：(任务数, 节点数)
# 修改这里会自动影响绘图顺序和标签，无需修改绘图代码
SCENARIOS = [
    (25, 150),
    (50, 100),
    (100, 50),
    (120, 30)
]

DATA_GEN_SEED = 2025
EXPERIMENT_SEED_BASE = 1000
NUM_REPEATS = 100
TS_ITERATIONS = 100
TS_PARAMS = {
    "tabu_tenure": 10,
    "neighborhood_size": 10,
    "diversification_threshold": 15
}

FORCE_RERUN = False


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# --- 3. 主逻辑 ---

def main():
    ensure_dir(OUTPUT_DIR)
    csv_path = os.path.join(OUTPUT_DIR, "latency_composition.csv")
    df = None

    print("=== Starting Latency Composition Analysis ===")

    # 尝试直接读取数据
    if os.path.exists(csv_path) and not FORCE_RERUN:
        print(f"\n[INFO] Found existing data at: {csv_path}")
        print("[INFO] Loading data and skipping simulation...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {e}. Will re-run simulation.")
            df = None

    if df is None:
        print(f"[INFO] Running simulations... (Force Rerun: {FORCE_RERUN})")
        final_results = []

        for task_num, node_num in SCENARIOS:
            scenario_name = f"{task_num}T / {node_num}N"
            print(f"\n--- Processing Scenario: {scenario_name} ---")

            # 3.1 数据生成
            scenario_file_prefix = f"T{task_num}_N{node_num}"
            task_info_path = os.path.join(OUTPUT_DIR, f"{scenario_file_prefix}_task.json")
            resource_info_path = os.path.join(OUTPUT_DIR, f"{scenario_file_prefix}_res.json")

            if not os.path.exists(task_info_path) or not os.path.exists(resource_info_path):
                print("  Generating scenario data...")
                set_seed(DATA_GEN_SEED)
                try:
                    task_dag = TaskDAG(num_of_task=task_num, random_seed=DATA_GEN_SEED)
                    task_dag.generate_dag()
                    task_dag.save_output(task_info_path)

                    net_gen = NetworkGenerator(num_of_compute_node=node_num, random_seed=DATA_GEN_SEED)
                    net_gen.generate_network(task_info_path=task_info_path, output_path=resource_info_path)
                except Exception as e:
                    print(f"Error generating data: {e}")
                    continue

            # 3.2 加载仿真器
            simulator = TaskFlowSimulator()
            simulator.load_resource_info(resource_info_path)
            simulator.load_task_info(task_info_path)

            metrics_accumulator = {'Comm': [], 'Comp': [], 'Queue': []}
            start_time = time.time()

            # 3.3 重复实验
            for i in range(NUM_REPEATS):
                run_seed = EXPERIMENT_SEED_BASE + task_num * 1000 + i
                set_seed(run_seed)

                ts = TabuSearch(
                    simulator=simulator,
                    num_tasks=task_num,
                    num_nodes=node_num,
                    max_iterations=TS_ITERATIONS,
                    **TS_PARAMS
                )
                best_allocation, best_priority = ts.run()

                # 3.4 回放最优解
                allocation_matrix = [[0] * task_num for _ in range(node_num)]
                for t_id, n_id in enumerate(best_allocation):
                    allocation_matrix[n_id][t_id] = 1

                simulator.task_allocation_info = {
                    "allocation_matrix": allocation_matrix,
                    "task_priorities": best_priority
                }

                latencies = simulator.simulate_key_task_flow_latency()

                run_total_comm = 0
                run_total_comp = 0
                run_total_queue = 0
                num_flows = len(latencies)

                for flow_id, l_info in latencies.items():
                    run_total_comm += l_info['total_package_latency'] + l_info['total_transfer_latency']
                    run_total_comp += l_info['total_compute_latency']
                    run_total_queue += l_info['total_task_queueing_latency']

                if num_flows > 0:
                    metrics_accumulator['Comm'].append(run_total_comm / num_flows)
                    metrics_accumulator['Comp'].append(run_total_comp / num_flows)
                    metrics_accumulator['Queue'].append(run_total_queue / num_flows)

                if (i + 1) % 20 == 0:
                    print(f"    Run {i + 1}/{NUM_REPEATS} completed...")

            duration = time.time() - start_time
            print(f"  Scenario completed in {duration:.2f}s")

            # 3.5 计算平均值
            avg_comm = np.mean(metrics_accumulator['Comm'])
            avg_comp = np.mean(metrics_accumulator['Comp'])
            avg_queue = np.mean(metrics_accumulator['Queue'])

            final_results.append({
                'Scenario': scenario_name,
                'Communication': avg_comm,
                'Computation': avg_comp,
                'Queueing': avg_queue,
                'Total': avg_comm + avg_comp + avg_queue
            })

        df = pd.DataFrame(final_results)
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Simulation finished. Data saved to {csv_path}")

    # --- 4. 绘图部分 ---
    print("\n--- Plotting Results ---")

    # 4.1 动态排序逻辑 (根据 SCENARIOS 配置自动排序)
    # 构造从 "XXT / YYN" 到 索引 的映射字典
    scenario_to_index = {f"{t}T / {n}N": i for i, (t, n) in enumerate(SCENARIOS)}

    # 过滤掉 CSV 中可能存在的、但当前 SCENARIOS 配置中没有的数据
    df = df[df['Scenario'].isin(scenario_to_index.keys())].copy()

    # 创建排序键并排序
    df['sort_key'] = df['Scenario'].map(scenario_to_index)
    df = df.sort_values('sort_key').reset_index(drop=True)

    print("Sorted Data for Plotting:")
    print(df[['Scenario', 'Total', 'Communication', 'Computation', 'Queueing']])

    # 4.2 准备绘图数据
    fig, ax = plt.subplots()  # figsize 已在上面 update 中统一设置

    # 生成 x 轴标签 (将 "25T / 150N" 转换为 LaTeX 格式 "$n=25, m=150$")
    x_labels = []
    for s in df['Scenario']:
        # 简单解析字符串
        t_str, n_str = s.replace('T', '').replace('N', '').split(' / ')
        # 使用 LaTeX 格式，\n 换行
        label = f"$n={t_str}$\n$m={n_str}$"
        x_labels.append(label)

    scenarios = range(len(df))  # 使用数字索引作为 x 轴
    comm = df['Communication']
    comp = df['Computation']
    queue = df['Queueing']

    bar_width = 0.6  # 稍微宽一点点，因为图变窄了

    colors = {
        'Comm': '#7293CB',
        'Comp': '#E1974C',
        'Queue': '#84BA5B'
    }

    # 绘制
    ax.bar(scenarios, comm, width=bar_width, label='Communication',
           color=colors['Comm'], alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.bar(scenarios, comp, width=bar_width, bottom=comm, label='Computation',
           color=colors['Comp'], alpha=0.9, edgecolor='black', linewidth=0.5)
    ax.bar(scenarios, queue, width=bar_width, bottom=comm + comp, label='Queueing',
           color=colors['Queue'], alpha=0.9, edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Average Total Latency (ms)')
    # ax.set_xlabel('Problem Scale') # 可以不写 Label，因为 x 轴刻度已经很清楚了
    # ax.set_title('Latency Composition Analysis')

    # 设置 X 轴刻度
    ax.set_xticks(scenarios)
    ax.set_xticklabels(x_labels)

    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 4.3 标注百分比 (优化阈值)
    for i in range(len(df)):
        total = df.loc[i, 'Total']

        def add_label(val, bottom_val, color='white'):
            if val > 0.04 * total:
                pct = (val / total) * 100
                y_pos = bottom_val + val / 2
                ax.text(i, y_pos, f'{pct:.1f}%', ha='center', va='center',
                        color=color, fontsize=10, fontweight='bold')

        add_label(df.loc[i, 'Communication'], 0)
        add_label(df.loc[i, 'Computation'], df.loc[i, 'Communication'])
        add_label(df.loc[i, 'Queueing'], df.loc[i, 'Communication'] + df.loc[i, 'Computation'])

    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "latency_composition_stacked_bar.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()