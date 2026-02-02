import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import random
import os

from matplotlib import ticker
import scipy.stats as stats

from Optimization_Methods.GA_Algorithms import GeneticAlgorithm
from Optimization_Methods.PSO_Algorithms import ParticleSwarmOptimization
from Optimization_Methods.SA_Algorithms import SimulatedAnnealing
from Optimization_Methods.TS_Algorithms import TabuSearch
from Optimization_Methods.RS_Algorithms import RandomSearch
from TTFM_Simulation_main import TaskFlowSimulator

# --- 绘图样式设置 (保持与你提供的脚本风格一致) ---
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (6, 4)
})


def run_experiment_and_get_final_values(algorithm_class, simulator, num_tasks, num_nodes,
                                        algorithm_params, num_runs=50, seed_start=42):
    """
    运行指定算法多次，并仅返回每次运行的最终最优适应度值
    """
    final_fitness_values = []

    print(f"Running {algorithm_class.__name__} for {num_runs} runs...")

    for run in range(num_runs):
        # 设置随机种子保证可复现性
        current_seed = seed_start + run
        random.seed(current_seed)
        np.random.seed(current_seed)

        # 实例化并运行算法
        algorithm = algorithm_class(simulator=simulator, num_tasks=num_tasks, num_nodes=num_nodes, **algorithm_params)
        best_solution = algorithm.run()

        # 获取该次运行找到的全局最优值
        # 注意：不同算法记录在 log_data 的最后一行或 best_fitness 属性中
        # 这里统一使用算法实例的 best_fitness 属性
        if hasattr(algorithm, 'best_fitness'):
            final_fitness_values.append(algorithm.best_fitness)
        else:
            # 如果属性不存在，尝试从 log 中读取
            final_fitness_values.append(algorithm.log_data[-1]['Best_Fitness_Global'])

    return np.array(final_fitness_values)


def compute_performance_ratios(results_dict):
    """
    计算性能比率 r_{s,p}
    r_{s,p} = t_{s,p} / min(t_{all, p})
    但在单问题多运行场景下，通常定义：
    r_{s,i} = t_{s,i} / t_{best_known}
    其中 t_{best_known} 是所有算法所有运行中找到的绝对最小值。
    """
    # 1. 找到绝对最小值 (Best Known Solution)
    all_values = []
    for algo_name, values in results_dict.items():
        all_values.extend(values)

    min_val = np.min(all_values)

    # 防止除以0（虽然时延通常 > 0）
    if min_val <= 1e-9:
        min_val = 1e-9
        print("Warning: Minimum fitness is close to 0, using epsilon for ratio calculation.")

    print(f"Global Best Fitness Found across all runs: {min_val}")

    ratios_dict = {}
    for algo_name, values in results_dict.items():
        # 计算比率：当前值 / 最小值
        # 值为1表示达到了最优，值越大表示性能越差
        ratios = values / min_val
        ratios_dict[algo_name] = ratios

    return ratios_dict


def plot_performance_profile(ratios_dict, configs, save_path=None, x_max=None):
    """
    绘制 Dolan-More 性能概况图
    修改：使用【手动指定 Z-order】来彻底解决遮挡问题
    """
    plt.figure(dpi=300)

    color_map = {cfg['name']: cfg['color'] for cfg in configs}
    # 线型定义
    line_styles = ['-']

    # --- 【关键修改】手动指定绘制层级 ---
    # 数值越大，画得越晚（越在顶层）
    # 你的需求是：GA 把 PSO 盖住了 -> 所以 PSO 的值必须比 GA 大
    manual_zorder = {
        'TS': 10,
        'RS': 20,
        'SA': 30,
        'GA': 40,
        'PSO': 50
    }

    # --- 1. 自动计算范围 ---
    all_ratios_concat = np.concatenate(list(ratios_dict.values()))
    max_ratio_in_data = np.max(all_ratios_concat)

    if x_max is None:
        calculated_x_max = max(1.05, max_ratio_in_data * 1.02)
    else:
        calculated_x_max = x_max

    # --- 2. 绘图循环 ---
    # 这里我们不需要对循环顺序排序了，因为 zorder 参数会控制覆盖关系
    for idx, (algo_name, ratios) in enumerate(ratios_dict.items()):
        sorted_ratios = np.sort(ratios)
        n = len(sorted_ratios)
        y_values = np.arange(1, n + 1) / n

        plot_x = np.concatenate(([1.0], sorted_ratios, [calculated_x_max]))
        plot_y = np.concatenate(([0.0], y_values, [y_values[-1]]))

        color = color_map.get(algo_name, 'black')
        linestyle = line_styles[idx % len(line_styles)]

        # 获取该算法的手动 Z-order，如果没写就默认 1
        z = manual_zorder.get(algo_name, 1)

        # 绘制
        plt.step(plot_x, plot_y, label=algo_name, color=color, linestyle=linestyle,
                 where='post', linewidth=1.5, alpha=0.9, zorder=z)  # <--- 传入 zorder

    # --- 3. 设置坐标轴 ---
    plt.xlabel(r'Performance Ratio ($\theta$)')
    plt.ylabel(r'Performance Profile $\rho_s(\theta)$')
    # plt.title('Dolan-Moré Performance Profile') 期刊要求图片里不要标题
    plt.xscale('linear')

    plt.xlim(1.0, calculated_x_max)
    plt.ylim(0.0, 1.02)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # 刻度密度设置
    if calculated_x_max - 1.0 < 0.2:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))
    elif calculated_x_max - 1.0 < 1.0:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    plt.grid(True, linestyle='--', alpha=0.6)

    # 图例顺序：通常希望图例顺序和 Z-order 或者 性能优劣 一致
    # 这里我们让 Matplotlib 自动处理，或者你可以手动排序 handles
    # 为了美观，我们按照 Friedman 排名顺序显示图例 (TS 在最上面)
    handles, labels = ax.get_legend_handles_labels()
    # 自定义排序列表
    desired_order = ['TS', 'RS', 'SA', 'PSO', 'GA']
    # 重新组织 handles 和 labels
    ordered_handles = []
    ordered_labels = []
    for algo in desired_order:
        if algo in labels:
            idx = labels.index(algo)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])

    plt.legend(ordered_handles, ordered_labels, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def perform_statistical_tests(df, output_dir):
    """
    执行 Friedman 检验和 Holm 后置检验，并保存 CSV 表格
    """
    print("\n" + "=" * 40)
    print("Starting Statistical Analysis (Friedman + Holm)")
    print("=" * 40)

    # 1. 转换数据为 Rank (数值越小 Rank 越小，Rank 1 = Best)
    # axis=1 表示在每一行(每次运行)内部对不同算法进行排名
    ranks = df.rank(axis=1, method='average', ascending=True)

    # 计算平均排名
    mean_ranks = ranks.mean()
    print("Average Ranks:")
    print(mean_ranks.sort_values())

    # 2. Friedman Test
    # 提取每一列的数据作为输入
    args = [df[col] for col in df.columns]
    statistic, p_value = stats.friedmanchisquare(*args)

    print(f"\nFriedman Test Result: Statistic={statistic:.4f}, p-value={p_value:.4e}")

    # 保存 Friedman 结果 (表格 1)
    friedman_res = pd.DataFrame({
        'Algorithm': mean_ranks.index,
        'Average_Rank': mean_ranks.values
    }).sort_values('Average_Rank')

    # 将检验统计量加到表格底部或作为元数据
    # 这里我们简单粗暴地加两列显示总体验证结果，或者单独存一个文件
    # 为了方便，这里把 p-value 放在文件名里，或者存为单独的 Summary
    summary_file = os.path.join(output_dir, "Stat_Friedman_Rankings.csv")

    # 添加一行显示 P-value
    with open(summary_file, 'w', newline='') as f:
        f.write(f"# Friedman Test: Chi2={statistic:.4f}, p-value={p_value:.6e}\n")
        friedman_res.to_csv(f, index=False)
    print(f"Friedman table saved to {summary_file}")

    # 如果 p-value > 0.05，通常不进行后置检验，但为了输出表格，我们强制执行

    # 3. Holm's Post-hoc Test
    # 找出 Control (Rank 最小的算法)
    control_algo = mean_ranks.idxmin()
    control_rank = mean_ranks.min()
    print(f"\nControl Algorithm (Best Rank): {control_algo}")

    k = len(df.columns)  # 算法数量
    n = len(df)  # 样本数 (Runs)

    # 计算标准误 SE
    se = np.sqrt((k * (k + 1)) / (6 * n))

    comparisons = []

    for algo in df.columns:
        if algo == control_algo:
            continue

        # 计算 Z 值
        # Z = (Rank_i - Rank_control) / SE
        # 注意：这里我们只关心是否有差异，取绝对值做双尾检验，或者根据方向
        diff = mean_ranks[algo] - control_rank
        z_value = diff / se

        # 计算未校正的 p 值 (双尾)
        # p = 2 * (1 - CDF(|z|))
        p_raw = 2 * (1 - stats.norm.cdf(abs(z_value)))

        comparisons.append({
            'Comparison': f"{control_algo} vs {algo}",
            'Algorithm': algo,
            'Z_value': z_value,
            'p_raw': p_raw
        })

    # 转换为 DataFrame 进行 Holm 校正
    holm_df = pd.DataFrame(comparisons)

    # 按 p_raw 从小到大排序
    holm_df = holm_df.sort_values('p_raw')

    # Holm Step-down Correction
    # p_adj = min(1, p_raw * (k - i))，且要保证单调性
    m = len(holm_df)  # 比较次数 = k - 1
    adj_p_values = []

    for i, row in enumerate(holm_df.itertuples()):
        # Holm 公式: p * (m - i)
        # 注意：i 从 0 开始，所以它是第 1 个 p 值，乘数是 m; 第 2 个乘 m-1
        multiplier = m - i
        p_adj = row.p_raw * multiplier

        # 修正 1：不能超过 1
        p_adj = min(1.0, p_adj)

        # 修正 2：单调性 (Adjusted p-value must be >= previous adjusted p-value)
        if i > 0:
            p_adj = max(p_adj, adj_p_values[-1])

        adj_p_values.append(p_adj)

    holm_df['p_Holm'] = adj_p_values

    # 格式化一下保留小数
    holm_df['Z_value'] = holm_df['Z_value'].map('{:.4f}'.format)
    holm_df['p_raw'] = holm_df['p_raw'].map('{:.6e}'.format)
    holm_df['p_Holm'] = holm_df['p_Holm'].map('{:.6e}'.format)

    # 判定显著性
    holm_df['Significant (alpha=0.05)'] = holm_df['p_Holm'].astype(float) < 0.05

    # 保存 Holm 结果 (表格 2)
    holm_file = os.path.join(output_dir, "Stat_Holm_Test.csv")
    holm_df.to_csv(holm_file, index=False)
    print(f"Holm test table saved to {holm_file}")


if __name__ == "__main__":
    # --- 参数配置 ---
    NUM_RUNS = 50  # 每个算法运行次数
    ITERATION_NUM = 100  # 算法内部迭代次数
    SEARCH_NUM = 10  # 种群大小/粒子数等
    MASTER_SEED = 2026

    OUTPUT_DIR = "result"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CSV_FILE = os.path.join(OUTPUT_DIR, "dmpp_final_fitness_data.csv")

    # --- 模拟器初始化 ---
    # 注意：请确保 JSON 路径正确，这里使用了相对路径 ../../TTFM_data/
    # 假设脚本在 Experiment/exp1... 中
    simulator = TaskFlowSimulator()
    try:
        simulator.load_resource_info("../../TTFM_data/computing_network_info.json")
        simulator.load_task_info("../../TTFM_data/task_info.json")
    except FileNotFoundError:
        # 尝试使用本地路径（如果用户将数据放在同级目录）
        print("Standard path failed, trying local path...")
        simulator.load_resource_info("TTFM_data/computing_network_info.json")
        simulator.load_task_info("TTFM_data/task_info.json")

    num_nodes = len(simulator.resource_info["compute_power"])
    num_tasks = len(simulator.task_info["task_node_values"])

    # --- 算法配置 (与 Draw_all_global_fitness.py 保持一致) ---
    algorithm_configs = [
        {
            "class": GeneticAlgorithm,
            "params": {"population_size": SEARCH_NUM, "max_generations": ITERATION_NUM, "crossover_rate": 0.6,
                       "mutation_rate": 0.1},
            "name": "GA",
            "color": "#19A102"
        },
        {
            "class": ParticleSwarmOptimization,
            "params": {"num_particles": SEARCH_NUM, "max_iterations": ITERATION_NUM},
            "name": "PSO",
            "color": "#303030"
        },
        {
            "class": SimulatedAnnealing,
            "params": {"max_iter_T": SEARCH_NUM, "num_T_changes": ITERATION_NUM, "T0": 100, "Tmin": 0.01},
            "name": "SA",
            "color": "#005ACF"
        },
        {
            "class": TabuSearch,
            "params": {"max_iterations": ITERATION_NUM, "tabu_tenure": SEARCH_NUM, "neighborhood_size": SEARCH_NUM,
                       "diversification_threshold": SEARCH_NUM},
            "name": "TS",
            "color": "#B10000"
        },
        {
            "class": RandomSearch,
            "params": {"max_iterations": ITERATION_NUM, "samples_per_iteration": SEARCH_NUM},
            "name": "RS",
            "color": "#FADB62"
        }
    ]

    # --- 执行或加载数据 ---
    results = {}

    # 检查是否已有缓存数据，避免重复运行
    if os.path.exists(CSV_FILE):
        print(f"Loading existing data from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE)
        for config in algorithm_configs:
            name = config['name']
            if name in df.columns:
                results[name] = df[name].values
            else:
                print(f"Warning: {name} not found in CSV.")
    else:
        print("No cached data found. Starting simulations...")
        df_data = {}
        for config in algorithm_configs:
            fitness_values = run_experiment_and_get_final_values(
                config["class"], simulator, num_tasks, num_nodes,
                config["params"], num_runs=NUM_RUNS, seed_start=MASTER_SEED
            )
            results[config["name"]] = fitness_values
            df_data[config["name"]] = fitness_values

        # 保存数据
        df = pd.DataFrame(df_data)
        df.to_csv(CSV_FILE, index=False)
        print(f"Data saved to {CSV_FILE}")

    # --- 数据处理 ---
    # 计算 DMPP 比率
    ratios = compute_performance_ratios(results)

    # --- 绘图 ---
    save_png_path = os.path.join(OUTPUT_DIR, "Dolan_More_Performance_Profile.png")

    # 你可以调整 x_max 来控制X轴显示的范围 (例如显示到最差算法的5倍)
    plot_performance_profile(ratios, algorithm_configs, save_path=save_png_path, x_max=1.8)

    perform_statistical_tests(df, OUTPUT_DIR)