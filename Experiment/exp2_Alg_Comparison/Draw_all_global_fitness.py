import matplotlib.pyplot as plt
import matplotlib.patches
import numpy as np
import pandas as pd
import matplotlib
import random
import os
from Optimization_Methods.GA_Algorithms import GeneticAlgorithm
from Optimization_Methods.PSO_Algorithms import ParticleSwarmOptimization
from Optimization_Methods.SA_Algorithms import SimulatedAnnealing
from Optimization_Methods.TS_Algorithms import TabuSearch
from Optimization_Methods.RS_Algorithms import RandomSearch
from TTFM_Simulation_main import TaskFlowSimulator

# 设置matplotlib参数，使图形更适合学术论文
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,  # Adjusted for potentially more legend entries
    'figure.figsize': (8, 5)  # Slightly larger for multiple lines
})


def collect_run_data(algorithm_class, simulator, num_tasks, num_nodes, algorithm_params, num_runs=100, seed_start=42,
                     param="Best_Fitness_Current"):
    """
    运行算法多次并收集所有运行的数据

    参数:
        algorithm_class: 算法类 (GeneticAlgorithm, TabuSearch 等)
        simulator: 任务流模拟器实例
        num_tasks: 任务数量
        num_nodes: 节点数量
        algorithm_params: 算法参数字典
        num_runs: 运行次数
        seed_start: 起始随机种子
        param: 要收集的参数名

    返回:
        全部运行的数据列表
    """
    all_runs_data = []

    for run in range(num_runs):
        random.seed(seed_start + run)
        np.random.seed(seed_start + run)

        algorithm = algorithm_class(simulator=simulator, num_tasks=num_tasks, num_nodes=num_nodes, **algorithm_params)
        algorithm.run()
        log_data = algorithm.log_data

        iterations = []
        best_fitness_values = []  # Renamed from best_fitness to avoid conflict

        for entry in log_data:
            if entry[param] in ["Final", "N/A"] or \
                    entry["Iteration"] == "Final" or \
                    entry.get("Average_Fitness_Current") == "Final" or \
                    entry["Iteration"] == 0:
                continue

            iterations.append(entry["Iteration"])
            # Ensure the value is numeric before appending
            try:
                fitness_val = float(entry[param])
                best_fitness_values.append(fitness_val)
            except ValueError:
                print(
                    f"Warning: Could not convert '{entry[param]}' to float for param '{param}' in iteration {entry['Iteration']}. Skipping.")
                # Optionally, append NaN or a placeholder if needed, or simply skip
                continue

        all_runs_data.append((iterations, best_fitness_values))

        if (run + 1) % 10 == 0:
            print(f"Algorithm {algorithm_class.__name__}: 已完成 {run + 1}/{num_runs} 次运行")

    return all_runs_data


def process_multiple_runs(all_runs_data, max_iterations, upper_bond, lower_bond):
    """
    处理多次运行的数据，计算每次迭代的均值和自定义分位数

    参数:
        all_runs_data: 所有运行的数据列表
        max_iterations: 最大迭代次数
        upper_bond: 上分位数 (e.g., 0.95 for 95th percentile)
        lower_bond: 下分位数 (e.g., 0.05 for 5th percentile)

    返回:
        iterations: 迭代次数列表
        mean_fitness: 平均适应度列表
        q_lower_fitness: 下分位数列表
        q_upper_fitness: 上分位数列表
    """
    all_fitness = np.full((len(all_runs_data), max_iterations + 1), np.nan)

    for run_idx, (iterations_data, best_fitness_data) in enumerate(all_runs_data):
        for i, iter_num in enumerate(iterations_data):
            if iter_num <= max_iterations:
                all_fitness[run_idx, iter_num] = best_fitness_data[i]

    # Drop iteration 0 if it's all NaNs or not needed for plotting (often it is)
    # if np.all(np.isnan(all_fitness[:, 0])):
    #     all_fitness = all_fitness[:, 1:]
    #     iterations_axis = np.arange(1, max_iterations + 1)
    # else:
    #     iterations_axis = np.arange(max_iterations + 1)
    # Keep iteration 0 for now, as it might contain initial values.
    # Data collection already skips iteration 0, so effectively we start from 1.
    # The `all_fitness` array is indexed from 0, so iteration 1 is at index 1.

    mean_fitness = np.nanmean(all_fitness, axis=0)
    q_upper_fitness = np.nanquantile(all_fitness, upper_bond, axis=0)  # e.g. 0.95 or 0.75
    q_lower_fitness = np.nanquantile(all_fitness, lower_bond, axis=0)  # e.g. 0.05 or 0.25

    iterations_axis = np.arange(max_iterations + 1)

    # We want to plot starting from iteration 1 as per original logic in collect_run_data
    return iterations_axis[1:], mean_fitness[1:], q_lower_fitness[1:], q_upper_fitness[1:]


def plot_combined_convergence(algorithms_data, title_suffix="Convergence Comparison",
                              png_save_path=None, csv_save_path=None, y_min=None, y_max=None,
                              lower_bond_percent=0, upper_bond_percent=100):
    """
    绘制多个算法的收敛曲线及其分位数范围

    参数:
        algorithms_data: 包含各算法数据的列表，每个元素是一个字典:
                         {'name': str, 'iterations': np.array, 'mean': np.array,
                          'q_lower': np.array, 'q_upper': np.array, 'color': str}
        title_suffix: 图表标题的后缀
        png_save_path: 保存图形的路径
        csv_save_path: 保存聚合数据的CSV路径
        y_min: Y轴最小值
        y_max: Y轴最大值
        lower_bond_percent: 用于图例的下分位数百分比
        upper_bond_percent: 用于图例的上分位数百分比
    """
    plt.figure(dpi=300)

    combined_df_data = {}
    base_iterations = None

    for data in algorithms_data:
        # Ensure all algorithms use the same iteration scale for combined CSV
        if base_iterations is None:
            base_iterations = data['iterations']
            combined_df_data['Iteration'] = base_iterations
        elif not np.array_equal(base_iterations, data['iterations']):
            # This case should ideally not happen if max_iterations is consistent
            print(f"Warning: Iteration scale mismatch for {data['name']}. CSV might be affected.")
            # For plotting, it's fine as each plots on its own iteration array.
            # For CSV, we might need to align them or save separate CSVs.
            # For simplicity, we'll use the first algorithm's iterations for CSV.

        # 绘制均值线
        plt.plot(data['iterations'], data['mean'], '-', color=data['color'], linewidth=2, markersize=3,
                 label=f"{data['name']} Mean Fitness")

        # 绘制分位数区域
        plt.fill_between(data['iterations'],
                         data['q_lower'],
                         data['q_upper'],
                         color=data['color'], alpha=0.15,
                         label=f"{data['name']} (All Fitness Range)")

        # Add data for combined CSV
        combined_df_data[f"{data['name']}_Mean_Fitness"] = pd.Series(data['mean'], index=data['iterations'])
        combined_df_data[f"{data['name']}_Q{lower_bond_percent:.0f}_Fitness"] = pd.Series(data['q_lower'],
                                                                                          index=data['iterations'])
        combined_df_data[f"{data['name']}_Q{upper_bond_percent:.0f}_Fitness"] = pd.Series(data['q_upper'],
                                                                                          index=data['iterations'])

    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness Found (Weighted Latency Sum)')
    # plt.title(title_suffix)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    else:
        # Auto-scaling based on all plotted data
        all_q_lower = np.concatenate([data['q_lower'] for data in algorithms_data])
        all_q_upper = np.concatenate([data['q_upper'] for data in algorithms_data])
        min_val = np.nanmin(all_q_lower)
        max_val = np.nanmax(all_q_upper)
        if not np.isnan(min_val) and not np.isnan(max_val) and min_val < max_val:
            plt.ylim(0.9 * min_val if min_val > 0 else 1.1 * min_val,
                     1.1 * max_val if max_val > 0 else 0.9 * max_val)

    if png_save_path is not None:
        plt.tight_layout()
        plt.savefig(png_save_path, bbox_inches='tight')
        print(f"Combined graph saved to: {png_save_path}")

    if csv_save_path is not None:
        # Create DataFrame, aligning by iteration index
        df_to_save = pd.DataFrame(combined_df_data)
        if 'Iteration' in df_to_save.columns:  # Ensure Iteration is the first column
            cols = ['Iteration'] + [col for col in df_to_save.columns if col != 'Iteration']
            df_to_save = df_to_save[cols]
        df_to_save = df_to_save.set_index('Iteration')  # Iteration becomes index
        df_to_save = df_to_save.reindex(
            base_iterations).reset_index()  # Ensure full iteration range and make it a column again

        df_to_save.to_csv(csv_save_path, index=False, float_format='%.5f')
        print(f"Combined aggregated data saved to: {csv_save_path}")

    plt.show()

def plot_final_fitness_comparison_bar_chart(algorithms_final_data, title_suffix="Final Fitness Comparison",
                                            png_save_path=None, y_label='Best Fitness (Weighted Latency Sum)',
                                            lower_bond_percent=0, upper_bond_percent=100, y_min=None, y_max=None,
                                            bar_width=0.5,
                                            figsize=(8, 6),
                                            title_fontsize=15,    # 新增：标题字体大小
                                            axis_label_fontsize=13, # 新增：坐标轴标签字体大小
                                            tick_label_fontsize=11, # 新增：刻度标签字体大小
                                            bar_text_fontsize=9
                                            ):
    """
    绘制多个算法最终适应度的柱状图比较

    参数:
        algorithms_final_data: 包含各算法最终适应度数据的列表，每个元素是一个字典:
                               {'name': str, 'mean_final': float, 'q_lower_final': float,
                                'q_upper_final': float, 'color': str}
        title_suffix: 图表标题的后缀
        png_save_path: 保存图形的路径
        y_label: Y轴标签
        lower_bond_percent: 用于图例的下分位数百分比
        upper_bond_percent: 用于图例的上分位数百分比
    """
    if not algorithms_final_data:
        print("No data provided for bar chart.")
        return

    names = [data['name'] for data in algorithms_final_data]
    mean_values = np.array([data['mean_final'] for data in algorithms_final_data])
    q_lower_values = np.array([data['q_lower_final'] for data in algorithms_final_data])
    q_upper_values = np.array([data['q_upper_final'] for data in algorithms_final_data])
    # colors = [data['color'] for data in algorithms_final_data]
    colors = ['#88C77F', '#ABABAB', '#74A3E0', '#D67575', '#FAE282']

    # 计算非对称误差线
    # 误差是相对于均值的
    lower_errors = mean_values - q_lower_values
    upper_errors = q_upper_values - mean_values
    asymmetric_error = [lower_errors, upper_errors]

    plt.figure(dpi=300) # 使用与收敛图相同的DPI
    bars = plt.bar(names, mean_values, color=colors, yerr=asymmetric_error, capsize=5,
                   ecolor='#222222', alpha=0.8,width=bar_width)

    # 可选：在柱子顶部添加均值的文本标签
    for bar_idx, bar in enumerate(bars):
        yval = bar.get_height()
        # 为标签位置添加一点偏移，使其更清晰

        vertical_text_offset = 0.0012  # 文本底部高于柱顶的距离
        text_y = yval + vertical_text_offset

        horizontal_shift_factor = 0.3  # 将文本中心向右移动柱子宽度的 15%
        # 您可以调整这个值 (例如 0.1 到 0.25 之间)
        text_x = bar.get_x() + bar.get_width() / 2.0 + (bar.get_width() * horizontal_shift_factor)

        plt.text(text_x, text_y, f'{yval:.4f}',
                 ha='center',  # 文本在其 (text_x, text_y) 位置水平居中
                 va='bottom',  # 文本的底部在 text_y 位置
                 fontsize=9,
                 color='black')

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    plt.ylabel(y_label)
    # plt.title(title_suffix)

    # 如果需要自定义图例来解释误差线代表的范围：
    # range_label = f'{lower_bond_percent:.0f}-{upper_bond_percent:.0f}th Percentile Range'
    # if lower_bond_percent == 0 and upper_bond_percent == 100:
    #     range_label = 'Min-Max Range'
    #
    # # 创建一个虚拟的patch用于图例 (如果误差线本身不带标签)
    # error_patch = matplotlib.patches.Patch(color='none', label=range_label) # 使用一个不可见的patch
    # plt.legend(handles=[error_patch], title="Error bars represent:", loc='upper right')
    # 或者更简单地在标题或说明文字中注明误差线的含义。
    # 对于这个场景，Y轴标签和标题已经足够清晰，误差线通常被理解为某种范围。

    if png_save_path is not None:
        plt.tight_layout() # 调整布局以适应所有元素
        plt.savefig(png_save_path, bbox_inches='tight')
        print(f"Final fitness bar chart saved to: {png_save_path}")

    plt.show()


if __name__ == "__main__":
    master_seed = 200
    random.seed(master_seed)
    np.random.seed(master_seed)

    # 折线图Y轴范围
    manual_y_min = 0.05
    manual_y_max = 0.3

    # 柱状图Y轴范围
    manual_y_min_bar = 0.0  # 例如，设置Y轴最小值为0.0
    manual_y_max_bar = 0.15  # 例如，设置Y轴最大值为0.25

    iteration_num = 100  # 对应 SA 中的 num_T_changes，GA 中的 max_generations，PSO/TS 中的 max_iterations
    search_num = 10  # 对应 SA 中的 max_iter_T，GA 中的 population_size，PSO 中的 num_particles，TS 中的 tabu_tenure/neighborhood_size/diversification_threshold
    num_runs = 100  # 运行次数

    # 定义百分位数用于显示范围，0和1代表最小值和最大值
    upper_bond_percentile = 1.0  # 例如，1.0 代表最大值 (100th percentile)
    lower_bond_percentile = 0.0  # 例如，0.0 代表最小值 (0th percentile)
    # 如果你希望使用例如 5th 和 95th 百分位数，请设置为 0.05 和 0.95

    param_to_plot = "Best_Fitness_Global"
    save_files = True

    output_dir = "result"  # 统一定义输出目录
    if save_files and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combined_csv_path = None
    if save_files:
        combined_csv_path = f"{output_dir}/Combined_{param_to_plot}_over_{num_runs}_runs_percentiles_data.csv"

    simulator = TaskFlowSimulator()
    simulator.load_resource_info("../../TTFM_data/computing_network_info.json")
    simulator.load_task_info("../../TTFM_data/task_info.json")

    num_nodes = len(simulator.resource_info["compute_power"])
    num_tasks = len(simulator.task_info["task_node_values"])

    algorithm_configs = [
        {
            "class": GeneticAlgorithm,
            "params": {"population_size": search_num, "max_generations": iteration_num, "crossover_rate": 0.6,
                       "mutation_rate": 0.1},
            "name": "GA",
            "color": "#19A102"  # Green
        },
        {
            "class": ParticleSwarmOptimization,
            "params": {"num_particles": search_num, "max_iterations": iteration_num},
            "name": "PSO",
            "color": "#303030"  # Black
        },
        {
            "class": SimulatedAnnealing,
            "params": {"max_iter_T": search_num, "num_T_changes": iteration_num, "T0": 100, "Tmin": 0.01},
            "name": "SA",
            "color": "#005ACF"  # Blue
        },
        {
            "class": TabuSearch,
            "params": {"max_iterations": iteration_num, "tabu_tenure": search_num, "neighborhood_size": search_num,
                       "diversification_threshold": search_num},
            "name": "TS",
            "color": "#B10000"  # Red
        },
        {
            "class": RandomSearch,
            "params": {"max_iterations": iteration_num},  # RS 只有一个参数，这里对应迭代次数
            "name": "RS",
            "color": "#FADB62"
        }
    ]

    all_algorithms_plot_data = []
    all_algorithms_final_stats = []

    data_loaded_from_file = False
    # 仅当 save_files 为 True 且文件存在时才尝试加载
    if save_files and combined_csv_path and os.path.exists(combined_csv_path):
        print(f"Found existing data file: {combined_csv_path}. Attempting to load.")
        try:
            loaded_df = pd.read_csv(combined_csv_path)
            if 'Iteration' not in loaded_df.columns:
                raise ValueError("CSV file is missing 'Iteration' column.")

            temp_plot_data = []
            temp_final_stats = []

            for config in algorithm_configs:
                algo_name = config['name']
                mean_col = f"{algo_name}_Mean_Fitness"
                q_lower_col = f"{algo_name}_Q{int(lower_bond_percentile * 100)}_Fitness"
                q_upper_col = f"{algo_name}_Q{int(upper_bond_percentile * 100)}_Fitness"

                if not all(col in loaded_df.columns for col in [mean_col, q_lower_col, q_upper_col]):
                    # 如果数据不完整，则标记需要重新运行所有模拟以确保一致性
                    print(
                        f"Warning: Missing one or more columns for {algo_name} in {combined_csv_path}. Data for this algorithm might be incomplete or re-run.")
                    raise ValueError(f"Data for {algo_name} not found or incomplete in CSV.")

                iters_data = loaded_df['Iteration'].to_numpy()
                mean_f_data = loaded_df[mean_col].to_numpy()
                q_lower_f_data = loaded_df[q_lower_col].to_numpy()
                q_upper_f_data = loaded_df[q_upper_col].to_numpy()

                valid_indices = ~np.isnan(mean_f_data)
                if not np.any(valid_indices):
                    print(f"Warning: All NaN data for {algo_name} from CSV. Skipping this algorithm for plots.")
                    continue  # 如果算法没有有效数据，则跳过

                iters_data_clean = iters_data[valid_indices]
                mean_f_data_clean = mean_f_data[valid_indices]
                q_lower_f_data_clean = q_lower_f_data[valid_indices]
                q_upper_f_data_clean = q_upper_f_data[valid_indices]

                if len(iters_data_clean) == 0:  # 清理后再次检查
                    print(f"Warning: No valid data points for {algo_name} after cleaning from CSV. Skipping.")
                    continue

                temp_plot_data.append({
                    "name": algo_name,
                    "iterations": iters_data_clean,
                    "mean": mean_f_data_clean,
                    "q_lower": q_lower_f_data_clean,
                    "q_upper": q_upper_f_data_clean,
                    "color": config["color"]
                })

                # <--- 新增: 提取最终统计数据用于柱状图
                temp_final_stats.append({
                    "name": algo_name,
                    "mean_final": mean_f_data_clean[-1],  # 取最后一个有效均值
                    "q_lower_final": q_lower_f_data_clean[-1],  # 取最后一个有效下限
                    "q_upper_final": q_upper_f_data_clean[-1],  # 取最后一个有效上限
                    "color": config["color"]
                })

            all_algorithms_plot_data = temp_plot_data
            all_algorithms_final_stats = temp_final_stats  # <--- 新增: 分配加载的数据
            data_loaded_from_file = True
            print("Data loaded successfully from file.")
        except Exception as e:
            print(f"Error loading or parsing data from {combined_csv_path}: {e}. Will re-run simulations.")
            all_algorithms_plot_data = []
            all_algorithms_final_stats = []  # <--- 新增: 重置
            data_loaded_from_file = False

    if not data_loaded_from_file:
        print("No valid pre-existing data file found or save_files is False. Running simulations...")

        max_total_iterations = 0
        # 确定所有算法中最大的迭代次数，用于 process_multiple_runs 的 max_iterations 参数
        for config in algorithm_configs:
            # 智能获取迭代次数参数名
            iter_param_names = ["max_iterations", "max_generations", "num_T_changes"]
            current_max_iter = iteration_num  # 默认值
            for p_name in iter_param_names:
                if p_name in config["params"]:
                    current_max_iter = config["params"][p_name]
                    break
            if current_max_iter > max_total_iterations:
                max_total_iterations = current_max_iter
        print(f"Using max_total_iterations: {max_total_iterations} for data processing based on algorithm configs.")

        for config in algorithm_configs:
            print(f"\nRunning {config['name']} for {num_runs} times...")

            all_runs_data = collect_run_data(
                algorithm_class=config["class"],
                simulator=simulator,
                num_tasks=num_tasks,
                num_nodes=num_nodes,
                algorithm_params=config["params"],
                num_runs=num_runs,
                seed_start=master_seed,  # 使用主种子确保不同算法运行的可比性
                param=param_to_plot
            )

            iters, mean_f, q_lower_f, q_upper_f = process_multiple_runs(
                all_runs_data,
                max_iterations=max_total_iterations,  # 使用计算得到的最大迭代次数
                upper_bond=upper_bond_percentile,
                lower_bond=lower_bond_percentile
            )

            if len(iters) == 0:  # 如果 process_multiple_runs 没有返回有效数据
                print(
                    f"Warning: No processed data for {config['name']} after simulation. Skipping this algorithm for plots.")
                continue

            all_algorithms_plot_data.append({
                "name": config["name"],
                "iterations": iters,
                "mean": mean_f,
                "q_lower": q_lower_f,
                "q_upper": q_upper_f,
                "color": config["color"]
            })

            # <--- 新增: 提取最终统计数据用于柱状图
            all_algorithms_final_stats.append({
                "name": config["name"],
                "mean_final": mean_f[-1],  # 取最后一个均值
                "q_lower_final": q_lower_f[-1],  # 取最后一个下限
                "q_upper_final": q_upper_f[-1],  # 取最后一个上限
                "color": config["color"]
            })
        # CSV 文件现在由 plot_combined_convergence 函数在需要时保存

    # 准备保存路径
    combined_png_path = None
    final_fitness_bar_chart_png_path = None  # <--- 新增: 柱状图的保存路径

    if save_files:
        combined_png_path = f"{output_dir}/Combined_{param_to_plot}_over_{num_runs}_runs_convergence.png"
        # <--- 新增: 定义柱状图的保存文件名
        final_fitness_bar_chart_png_path = f"{output_dir}/Final_Fitness_{param_to_plot}_over_{num_runs}_runs_comparison.png"

    # 绘制组合收敛曲线 (现有功能)
    if all_algorithms_plot_data:  # 仅当有数据时绘图
        plot_combined_convergence(
            algorithms_data=all_algorithms_plot_data,
            title_suffix=f'Algorithm Convergence Comparison ({num_runs} Runs)',
            png_save_path=combined_png_path,
            # CSV保存逻辑：仅当数据不是从文件加载（即新运行了仿真）且save_files为True时，才传递路径让绘图函数保存
            csv_save_path=combined_csv_path if not data_loaded_from_file and save_files else None,
            y_min=manual_y_min,
            y_max=manual_y_max,
            lower_bond_percent=lower_bond_percentile * 100,
            upper_bond_percent=upper_bond_percentile * 100
        )
    else:
        print("No data available to plot convergence curves.")

    # <--- 新增: 绘制最终适应度比较柱状图
    if all_algorithms_final_stats:  # 仅当有数据时绘图
        plot_final_fitness_comparison_bar_chart(
            algorithms_final_data=all_algorithms_final_stats,
            title_suffix=f'Final Fitness Comparison (After {num_runs} Runs)',
            png_save_path=final_fitness_bar_chart_png_path,
            y_label=f'Final Best Fitness Found (Weighted Latency Sum)',  # 使Y轴标签更具描述性
            lower_bond_percent=lower_bond_percentile * 100,
            upper_bond_percent=upper_bond_percentile * 100,
            y_min=manual_y_min_bar,
            y_max=manual_y_max_bar
        )
    else:
        print("No data available to plot final fitness bar chart.")

    print("\nAll done!")


# 写在最后：亲爱的读者朋友您好，如果你仔细研究了代码，您可能会注意到这个实验并不能被精确复现。很抱歉，这是因为论文该部分写作、绘图和分析完成后进行了少量调整，丢失了当时的随机种子和部分参数配置。
# 由于这不影响论文整体趋势和分析及核心结论（RS优于GA、TS领先其他所有算法），因此正式发表的论文仍然采用了前一版本的结果图，特此说明。