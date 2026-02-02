import random
import math
import csv
from TTFM_Simulation_main import TaskFlowSimulator

random.seed(114514)


class SimulatedAnnealing:
    def __init__(self, simulator, num_tasks, num_nodes, max_iter_T=10, num_T_changes=100, T0=100, Tmin=0.01):
        """
        初始化模拟退火算法参数
        """
        self.simulator = simulator
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.T0 = T0  # 初始温度
        self.T = T0  # 当前温度
        self.Tmin = Tmin  # 最低温度
        self.max_iter_T = max_iter_T  # 每个温度下的最大迭代次数

        # 根据温度变化次数计算温度衰减率 alpha
        self.alpha = (self.Tmin / self.T0) ** (1 / num_T_changes)

        # 随机初始化任务分配和优先级
        self.current_solution = self.random_solution()
        self.current_fitness = self.evaluate_fitness(self.current_solution)

        # 记录全局最优解
        self.best_solution = self.current_solution
        self.best_fitness = self.current_fitness

        # 日志记录
        self.log_data = []
        self.log_headers = ["Iteration", "Temperature", "Best_Fitness_Current", "Best_Fitness_Global",
                            "Acceptance_Rate", "Best_Solution_Changed"]
        self.accepted_count = 0
        self.total_count = 0

    def random_solution(self):
        """
        生成随机任务分配和优先级
        """
        allocation = [random.randint(0, self.num_nodes - 1) for _ in range(self.num_tasks)]
        priority = list(range(1, self.num_tasks + 1))
        random.shuffle(priority)
        return allocation, priority

    def evaluate_fitness(self, solution):
        """
        计算方案的适应度（关键任务流的平均总时延）
        """
        allocation_matrix = [[0] * self.num_tasks for _ in range(self.num_nodes)]
        for task_id, node_id in enumerate(solution[0]):
            allocation_matrix[node_id][task_id] = 1

        # 设置任务分配和优先级
        self.simulator.task_allocation_info = {
            "allocation_matrix": allocation_matrix,
            "task_priorities": solution[1]
        }

        # 获取时延信息
        latencies = self.simulator.simulate_key_task_flow_latency()
        total_latency = sum(latencies[flow_id].get("total_latency", 0) for flow_id in latencies)
        count = len(latencies)

        return total_latency / count if count > 0 else float('inf')

    def perturb_solution(self, solution):
        """
        产生新解：随机调整任务分配或交换任务优先级
        """
        new_allocation = solution[0][:]
        new_priority = solution[1][:]

        if random.random() < 0.5:  # 50% 概率调整任务分配
            task_id = random.randint(0, self.num_tasks - 1)
            new_allocation[task_id] = random.randint(0, self.num_nodes - 1)
        else:  # 50% 概率交换任务优先级
            task1, task2 = random.sample(range(self.num_tasks), 2)
            new_priority[task1], new_priority[task2] = new_priority[task2], new_priority[task1]

        return new_allocation, new_priority

    def acceptance_probability(self, old_fitness, new_fitness):
        """
        计算接受概率，如果新解更优则直接接受，否则以一定概率接受较差解
        """
        if new_fitness < old_fitness:
            return 1.0  # 更优解直接接受
        else:
            return math.exp((old_fitness - new_fitness) / self.T)  # 以一定概率接受劣解

    def run(self):
        """
        执行模拟退火算法
        """
        iteration = 0

        # 初始记录
        self.log_data.append({
            "Iteration": 0,
            "Temperature": self.T,
            "Best_Fitness_Current": self.current_fitness,
            "Best_Fitness_Global": self.best_fitness,
            "Acceptance_Rate": 1.0,  # 初始接受率设为1.0
            "Best_Solution_Changed": 1
        })

        while self.T > self.Tmin:
            self.accepted_count = 0  # 当前温度下接受的新解数量
            self.total_count = 0  # 当前温度下总尝试次数

            for _ in range(self.max_iter_T):
                self.total_count += 1

                # 生成新解
                new_solution = self.perturb_solution(self.current_solution)
                new_fitness = self.evaluate_fitness(new_solution)

                # 判断是否接受新解
                accept_prob = self.acceptance_probability(self.current_fitness, new_fitness)
                solution_improved = False

                if random.random() < accept_prob:
                    self.current_solution = new_solution
                    self.current_fitness = new_fitness
                    self.accepted_count += 1

                    # 更新全局最优解
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_solution
                        self.best_fitness = new_fitness
                        solution_improved = True

            # 添加日志
            acceptance_rate = self.accepted_count / self.total_count if self.total_count > 0 else 0

            self.log_data.append({
                "Iteration": iteration + 1,
                "Temperature": self.T,
                "Best_Fitness_Current": self.current_fitness,
                "Best_Fitness_Global": self.best_fitness,
                "Acceptance_Rate": acceptance_rate,
                "Best_Solution_Changed": int(solution_improved)
            })

            # 退火（降低温度）
            self.T *= self.alpha
            iteration += 1

        # 记录最终结果
        self.log_data.append({
            "Iteration": iteration,
            "Temperature": "Final",
            "Best_Fitness_Current": self.current_fitness,
            "Best_Fitness_Global": self.best_fitness,
            "Acceptance_Rate": "N/A",
            "Best_Solution_Changed": "N/A"
        })

        return self.best_solution

    def save_logs_to_csv(self, filename="sa_logs.csv"):
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
    simulator = TaskFlowSimulator()
    simulator.load_resource_info('../TTFM_data/computing_network_info.json')
    simulator.load_task_info('../TTFM_data/task_info.json')

    num_nodes = len(simulator.resource_info["compute_power"])
    num_tasks = len(simulator.task_info["task_node_values"])

    sa = SimulatedAnnealing(
        simulator, num_tasks, num_nodes,
        max_iter=10,
        T0=100,
        Tmin=0.01,
        alpha=0.912
    )
    best_allocation, best_priority = sa.run()

    # 保存日志到CSV
    sa.save_logs_to_csv("output/sa_log.csv")

    # 打印日志数据
    print("Simulated Annealing Algorithm Logs:")
    for header in sa.log_headers:
        print(f"{header}", end=",")
    print()
    for row in sa.log_data:
        for header in sa.log_headers:
            print(f"{row[header]}", end=",")
        print()

    # 输出最优分配方案
    allocation_matrix = [[0] * num_tasks for _ in range(num_nodes)]
    for task_id, node_id in enumerate(best_allocation):
        allocation_matrix[node_id][task_id] = 1

    print("\nBest Allocation Matrix:")
    for row in allocation_matrix:
        print(row)
    print("Best Priority Vector:", best_priority)
