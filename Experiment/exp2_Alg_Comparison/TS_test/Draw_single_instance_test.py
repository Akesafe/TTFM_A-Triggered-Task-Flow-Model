import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

import random

from Optimization_Methods.GA_Algorithms import GeneticAlgorithm
from Optimization_Methods.PSO_Algorithms import ParticleSwarmOptimization
from Optimization_Methods.SA_Algorithms import SimulatedAnnealing
from Optimization_Methods.TS_Algorithms import TabuSearch
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
    'legend.fontsize': 12,
    'figure.figsize': (6, 4)
})


def plot_convergence(result_instance, save_path=None, save_path_csv=None, algorithm_name='GA', ADD_SMOOTHLINE=False):
    """
    绘制遗传算法的收敛曲线 (迭代次数 vs 最佳适应度)

    参数:
        result_instance: 算法实例
        save_path: 保存图形的路径
    """

    log_data = result_instance.log_data
    # 提取迭代次数和最佳适应度
    iterations = []
    best_fitness = []

    for entry in log_data:
        # 跳过标记为"Final"或者"N/A",以及Iteration为Final的条目
        if entry["Best_Fitness_Current"] in ["Final", "N/A"]:
            continue
        if entry["Iteration"] == "Final":
            continue
        if entry.get("Average_Fitness_Current") == "Final":
            continue
        iterations.append(entry["Iteration"])
        best_fitness.append(entry["Best_Fitness_Current"])

    # 创建图形
    plt.figure(dpi=300)
    plt.plot(iterations, best_fitness, 'o-', color='#1f77b4', linewidth=2, markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness (Weighted Latency Sum)')
    plt.title(f'Convergence of {algorithm_name} (Single instance)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加平滑曲线以突出趋势
    if len(iterations) > 5 and ADD_SMOOTHLINE:  # 只有当有足够的数据点时才添加平滑曲线
        x_smooth = np.linspace(min(iterations), max(iterations), 100)
        # 使用多项式拟合
        z = np.polyfit(iterations, best_fitness, 3)
        p = np.poly1d(z)
        plt.plot(x_smooth, p(x_smooth), '-', color='#ff7f0e', linewidth=1.5,
                 alpha=0.8, label='Trend')
        plt.legend()

    # 设置适当的y轴范围
    plt.ylim(0.9 * min(best_fitness), 1.1 * max(best_fitness))

    if save_path is not None:
        # 保存图形
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"图形已保存至: {save_path}")

    if save_path_csv is not None:
        data_to_save = pd.DataFrame({
            'Iteration': iterations,
            'Best_Fitness': best_fitness,
        })
        data_to_save.to_csv(save_path_csv, index=False)
        print(f"聚合数据已保存至: {save_path_csv}")

    # 显示图形
    plt.show()


if __name__ == "__main__":
    random.seed(114514)

    # 加载模拟器
    simulator = TaskFlowSimulator()
    simulator.load_resource_info("../TTFM_data/computing_network_info.json")
    simulator.load_task_info("../TTFM_data/task_info.json")

    # 获取任务和节点数量
    num_nodes = len(simulator.resource_info["compute_power"])
    num_tasks = len(simulator.task_info["task_node_values"])

    # 初始化各种算法
    ga = GeneticAlgorithm(
        simulator, num_tasks, num_nodes,
        population_size=10,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.2
    )
    ts = TabuSearch(
        simulator=simulator, num_tasks=num_tasks, num_nodes=num_nodes,
        max_iterations=100,
        tabu_tenure=10,
        neighborhood_size=10,
        diversification_threshold=10
    )
    sa = SimulatedAnnealing(
        simulator, num_tasks, num_nodes,
        max_iter_T=10,
        num_T_changes=100,
        T0=100,
        Tmin=0.01,
    )
    pso = ParticleSwarmOptimization(
        simulator, num_tasks, num_nodes,
        num_particles=10,
        max_iterations=100
    )

    # 选择一种算法，ga/ts/sa/pso
    algorithm='ts'

    if algorithm == 'ga':
        algorithm_to_chose=ga
        algorithm_name = "Genetic Algorithm"
    elif algorithm == 'ts':
        algorithm_to_chose=ts
        algorithm_name = "Tabu Search"
    elif algorithm == 'sa':
        algorithm_to_chose=sa
        algorithm_name = "Simulated Annealing"
    elif algorithm == 'pso':
        algorithm_to_chose=pso
        algorithm_name = "Particle Swarm Optimization"
    else:
        algorithm_to_chose=None


    # 绘制收敛曲线
    algorithm_to_chose.run()
    plot_convergence(result_instance=algorithm_to_chose,
                     save_path=f"output/{algorithm}_Best_Fitness_Current.png",
                     save_path_csv=f"output/{algorithm}_Best_Fitness_Current_data.csv",
                     algorithm_name=algorithm_name)

    print("完成!")
