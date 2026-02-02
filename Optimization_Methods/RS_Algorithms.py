import random
import csv
from TTFM_Simulation_main import TaskFlowSimulator

# 设定随机种子以保证可复现性
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)


class RandomSearch:
    def __init__(self, simulator, num_tasks, num_nodes, max_iterations=100, samples_per_iteration=10):
        """
        初始化随机搜索算法

        参数:
        - simulator: 任务流模拟器实例
        - num_tasks: 任务数量
        - num_nodes: 节点数量
        - max_iterations: 外层迭代次数 (对应 GA 的 Generation, PSO 的 Iteration)
        - samples_per_iteration: 每次迭代生成的样本数 (对应 GA 的 Population, PSO 的 Particles)
        """
        self.simulator = simulator
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.max_iterations = max_iterations
        self.samples_per_iteration = samples_per_iteration  # 新增参数

        # 最优解记录
        self.best_solution = None
        self.best_fitness = float('inf')

        # 日志记录
        self.log_data = []
        # Best_Fitness_Current 代表当前这一批样本中的最优值
        # Best_Fitness_Global 代表历史全局最优值
        self.log_headers = ["Iteration", "Best_Fitness_Current", "Best_Fitness_Global", "Best_Solution_Changed"]

    def generate_random_solution(self):
        """
        生成一个完全随机的解
        """
        # 随机分配任务到节点
        allocation = [random.randint(0, self.num_nodes - 1) for _ in range(self.num_tasks)]
        # 随机生成任务优先级向量 (1 到 num_tasks 的随机排列)
        priority = list(range(1, self.num_tasks + 1))
        random.shuffle(priority)

        return (allocation, priority)

    def evaluate_fitness(self, solution):
        """
        计算解的适应度（关键任务流的平均总时延）
        """
        allocation, priority = solution

        # 转换为分配矩阵格式
        allocation_matrix = [[0] * self.num_tasks for _ in range(self.num_nodes)]
        for task_id, node_id in enumerate(allocation):
            allocation_matrix[node_id][task_id] = 1

        # 更新模拟器的配置
        self.simulator.task_allocation_info = {
            "allocation_matrix": allocation_matrix,
            "task_priorities": priority
        }

        # 运行仿真获取时延
        latencies = self.simulator.simulate_key_task_flow_latency()

        # 计算平均总时延
        total_latency = sum(latency_dict.get("total_latency", 0) for latency_dict in latencies.values())
        count = len(latencies)

        return total_latency / count if count > 0 else float('inf')

    def run(self):
        """
        执行随机搜索主循环
        """

        # 初始解生成（可选，为了对应迭代0的状态）
        initial_sol = self.generate_random_solution()
        initial_fit = self.evaluate_fitness(initial_sol)
        self.best_solution = initial_sol
        self.best_fitness = initial_fit

        # 记录初始状态
        self.log_data.append({
            "Iteration": 0,
            "Best_Fitness_Current": initial_fit,
            "Best_Fitness_Global": self.best_fitness,
            "Best_Solution_Changed": 1
        })

        # 主循环
        for iteration in range(self.max_iterations):

            # 用于记录当前这一轮（batch）中的最优情况
            current_batch_best_fitness = float('inf')
            solution_improved_in_this_batch = 0

            # 内层循环：批量采样 (对应种群大小/粒子数/邻域大小)
            for _ in range(self.samples_per_iteration):
                # 1. 生成随机解
                candidate_solution = self.generate_random_solution()

                # 2. 计算适应度
                fitness = self.evaluate_fitness(candidate_solution)

                # 3. 更新当前批次的最优记录
                if fitness < current_batch_best_fitness:
                    current_batch_best_fitness = fitness

                # 4. 更新全局最优解
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = candidate_solution
                    solution_improved_in_this_batch = 1

            # 5. 记录日志 (每完成一批采样记录一次，与其他算法对齐)
            self.log_data.append({
                "Iteration": iteration + 1,
                "Best_Fitness_Current": current_batch_best_fitness,  # 这一轮里找到的最好的
                "Best_Fitness_Global": self.best_fitness,  # 历史上最好的
                "Best_Solution_Changed": solution_improved_in_this_batch
            })

        # 记录结束标记
        self.log_data.append({
            "Iteration": "Final",
            "Best_Fitness_Current": "N/A",
            "Best_Fitness_Global": self.best_fitness,
            "Best_Solution_Changed": "N/A"
        })

        return self.best_solution

    def save_logs_to_csv(self, filename="rs_logs.csv"):
        """
        将日志保存到CSV文件
        """
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.log_headers)
            writer.writeheader()
            for row in self.log_data:
                writer.writerow(row)


# 示例使用
if __name__ == "__main__":
    # 加载模拟器
    simulator = TaskFlowSimulator()
    # 假设路径结构与其他算法文件一致
    simulator.load_resource_info("../TTFM_data/computing_network_info.json")
    simulator.load_task_info("../TTFM_data/task_info.json")

    # 获取任务和节点数量
    num_nodes = len(simulator.resource_info["compute_power"])
    num_tasks = len(simulator.task_info["task_node_values"])

    # 配置随机搜索参数 (例如：搜索 1000 次)
    max_search_times = 1000

    # 实例化并运行
    rs = RandomSearch(
        simulator=simulator,
        num_tasks=num_tasks,
        num_nodes=num_nodes,
        max_iterations=max_search_times
    )

    print(f"Starting Random Search for {max_search_times} iterations...")
    best_solution = rs.run()

    # 保存日志
    output_csv = "output/rs_log.csv"
    # 确保 output 文件夹存在，或者直接保存到当前目录
    try:
        rs.save_logs_to_csv(output_csv)
        print(f"Logs saved to {output_csv}")
    except FileNotFoundError:
        rs.save_logs_to_csv("rs_log.csv")
        print("Logs saved to rs_log.csv")

    # 打印最终结果
    print("\n--- Final Best Solution ---")
    print(f"Best Fitness (Average Latency): {rs.best_fitness}")

    allocation, priority = best_solution

    # 还原并打印分配矩阵
    allocation_matrix = [[0] * num_tasks for _ in range(num_nodes)]
    for task_id, node_id in enumerate(allocation):
        allocation_matrix[node_id][task_id] = 1

    print("\nBest Allocation Matrix:")
    for row in allocation_matrix:
        print(row)
    print(f"Best Priority Vector: {priority}")