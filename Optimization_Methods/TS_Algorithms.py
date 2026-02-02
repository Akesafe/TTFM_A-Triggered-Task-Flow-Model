import random
import copy
import csv
from TTFM_Simulation_main import TaskFlowSimulator

# 设定随机种子
RANDOM_SEED = 114514
random.seed(RANDOM_SEED)


class TabuSearch:
    def __init__(self, simulator, num_tasks, num_nodes, max_iterations=100,
                 tabu_tenure=10, neighborhood_size=10, diversification_threshold=10):
        """
        初始化禁忌搜索算法的参数

        参数:
        - simulator: 任务流模拟器
        - num_tasks: 任务数量
        - num_nodes: 节点数量
        - max_iterations: 最大迭代次数
        - tabu_tenure: 禁忌期限（移动被禁止的迭代次数）
        - neighborhood_size: 每次迭代生成的邻域解数量
        - diversification_threshold: 触发多样化策略的迭代次数阈值
        """
        self.simulator = simulator
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size
        self.diversification_threshold = diversification_threshold

        # 禁忌表：记录最近禁止的移动
        self.tabu_list_allocation = {}  # {(task_id, old_node, new_node): expiration_iteration}
        self.tabu_list_priority = {}  # {(pos1, pos2): expiration_iteration}

        # 最优解记录
        self.best_solution = None
        self.best_fitness = float('inf')

        # 停滞计数器
        self.stagnation_counter = 0

        # 日志记录
        self.log_data = []
        self.log_headers = ["Iteration", "Best_Fitness_Current", "Best_Fitness_Global", "Tabu_List_Size",
                            "Diversification_Applied", "Best_Solution_Changed"]

    def generate_initial_solution(self):
        """
        生成初始解：随机分配任务到节点，随机生成任务优先级
        """
        # 随机分配任务到节点
        allocation = [random.randint(0, self.num_nodes - 1) for _ in range(self.num_tasks)]
        # 随机生成任务优先级
        priority = list(range(1, self.num_tasks + 1))
        random.shuffle(priority)

        return (allocation, priority)

    def evaluate_fitness(self, solution):
        """
        计算解的适应度（目标是最小化关键任务流的平均总时延）
        """
        allocation, priority = solution

        # 转换为分配矩阵
        allocation_matrix = [[0] * self.num_tasks for _ in range(self.num_nodes)]
        for task_id, node_id in enumerate(allocation):
            allocation_matrix[node_id][task_id] = 1

        # 设置任务分配和优先级
        self.simulator.task_allocation_info = {
            "allocation_matrix": allocation_matrix,
            "task_priorities": priority
        }

        # 获取延迟信息
        latencies = self.simulator.simulate_key_task_flow_latency()

        # 计算平均总时延
        total_latency = sum(latency_dict.get("total_latency", 0) for latency_dict in latencies.values())
        count = len(latencies)

        return total_latency / count if count > 0 else float('inf')

    def generate_neighbors(self, current_solution, iteration):
        """
        生成当前解的邻域解，并过滤掉禁忌移动
        """
        current_allocation, current_priority = current_solution
        neighbors = []

        for _ in range(self.neighborhood_size):
            # 决定是修改任务分配还是任务优先级
            if random.random() < 0.5:  # 50% 概率修改任务分配
                neighbor_allocation = current_allocation.copy()
                neighbor_priority = current_priority.copy()

                # 随机选择一个任务并重新分配到另一个节点
                task_id = random.randint(0, self.num_tasks - 1)
                old_node = neighbor_allocation[task_id]

                # 选择一个不同的节点
                new_node = old_node
                while new_node == old_node:
                    new_node = random.randint(0, self.num_nodes - 1)

                # 检查是否在禁忌表中
                move_key = (task_id, old_node, new_node)
                if move_key in self.tabu_list_allocation and self.tabu_list_allocation[move_key] > iteration:
                    # 特赦策略：如果这个移动可能导致全局最优解，允许该移动
                    temp_allocation = neighbor_allocation.copy()
                    temp_allocation[task_id] = new_node
                    temp_solution = (temp_allocation, neighbor_priority)
                    temp_fitness = self.evaluate_fitness(temp_solution)

                    if temp_fitness < self.best_fitness:
                        neighbor_allocation[task_id] = new_node
                        neighbors.append(
                            ((task_id, old_node, new_node), None, (neighbor_allocation, neighbor_priority)))
                else:
                    neighbor_allocation[task_id] = new_node
                    neighbors.append(((task_id, old_node, new_node), None, (neighbor_allocation, neighbor_priority)))

            else:  # 50% 概率修改任务优先级
                neighbor_allocation = current_allocation.copy()
                neighbor_priority = current_priority.copy()

                # 随机选择两个位置交换优先级
                pos1, pos2 = random.sample(range(self.num_tasks), 2)

                # 检查是否在禁忌表中
                move_key = (pos1, pos2)
                if move_key in self.tabu_list_priority and self.tabu_list_priority[move_key] > iteration:
                    # 特赦策略
                    temp_priority = neighbor_priority.copy()
                    temp_priority[pos1], temp_priority[pos2] = temp_priority[pos2], temp_priority[pos1]
                    temp_solution = (neighbor_allocation, temp_priority)
                    temp_fitness = self.evaluate_fitness(temp_solution)

                    if temp_fitness < self.best_fitness:
                        neighbor_priority[pos1], neighbor_priority[pos2] = neighbor_priority[pos2], neighbor_priority[
                            pos1]
                        neighbors.append((None, (pos1, pos2), (neighbor_allocation, neighbor_priority)))
                else:
                    neighbor_priority[pos1], neighbor_priority[pos2] = neighbor_priority[pos2], neighbor_priority[pos1]
                    neighbors.append((None, (pos1, pos2), (neighbor_allocation, neighbor_priority)))

        return neighbors

    def diversify(self, current_solution):
        """
        多样化策略：当搜索停滞时，生成一个与当前解有一定距离的新解
        """
        current_allocation, current_priority = current_solution
        new_allocation = current_allocation.copy()
        new_priority = current_priority.copy()

        # 显著改变任务分配（改变约1/3的任务分配）
        tasks_to_change = random.sample(range(self.num_tasks), max(1, self.num_tasks // 3))
        for task_id in tasks_to_change:
            new_allocation[task_id] = random.randint(0, self.num_nodes - 1)

        # 显著改变任务优先级（打乱1/3的优先级）
        positions_to_change = random.sample(range(self.num_tasks), max(1, self.num_tasks // 3))
        values_to_change = [new_priority[pos] for pos in positions_to_change]
        random.shuffle(values_to_change)
        for i, pos in enumerate(positions_to_change):
            new_priority[pos] = values_to_change[i]

        return (new_allocation, new_priority)

    def run(self):
        """
        执行禁忌搜索算法寻找最优解
        """
        # 生成初始解
        current_solution = self.generate_initial_solution()
        current_fitness = self.evaluate_fitness(current_solution)

        # 更新最优解
        self.best_solution = current_solution
        self.best_fitness = current_fitness

        # 记录初始日志
        self.log_data.append({
            "Iteration": 0,
            "Best_Fitness_Current": current_fitness,
            "Best_Fitness_Global": self.best_fitness,
            "Tabu_List_Size": 0,
            "Diversification_Applied": 0,
            "Best_Solution_Changed": 1
        })

        # 主循环
        for iteration in range(self.max_iterations):
            # 生成邻域解
            neighbors = self.generate_neighbors(current_solution, iteration)

            if not neighbors:
                # 记录日志
                self.log_data.append({
                    "Iteration": iteration + 1,
                    "Best_Fitness_Current": current_fitness,
                    "Best_Fitness_Global": self.best_fitness,
                    "Tabu_List_Size": len(self.tabu_list_allocation) + len(self.tabu_list_priority),
                    "Diversification_Applied": 0,
                    "Best_Solution_Changed": 0
                })
                print("Warning: No neighbors")
                continue

            # 评估邻域解并选择最佳移动
            best_neighbor = None
            best_neighbor_fitness = float('inf')
            best_neighbor_allocation_move = None
            best_neighbor_priority_move = None

            for allocation_move, priority_move, neighbor in neighbors:
                fitness = self.evaluate_fitness(neighbor)

                if fitness < best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = fitness
                    best_neighbor_allocation_move = allocation_move
                    best_neighbor_priority_move = priority_move

            # 更新当前解
            current_solution = best_neighbor
            current_fitness = best_neighbor_fitness

            # 更新禁忌表
            if best_neighbor_allocation_move:
                self.tabu_list_allocation[best_neighbor_allocation_move] = iteration + self.tabu_tenure

            if best_neighbor_priority_move:
                self.tabu_list_priority[best_neighbor_priority_move] = iteration + self.tabu_tenure

            # 是否需要多样化
            diversification_applied = 0
            solution_improved = 0

            # 更新最优解
            if current_fitness < self.best_fitness:
                self.best_solution = current_solution
                self.best_fitness = current_fitness
                self.stagnation_counter = 0
                solution_improved = 1
            else:
                self.stagnation_counter += 1

            # 应用多样化策略
            if self.stagnation_counter >= self.diversification_threshold:
                current_solution = self.diversify(current_solution)
                current_fitness = self.evaluate_fitness(current_solution)
                self.stagnation_counter = 0
                diversification_applied = 1

            # 清理过期的禁忌表条目
            self.tabu_list_allocation = {k: v for k, v in self.tabu_list_allocation.items() if v > iteration}
            self.tabu_list_priority = {k: v for k, v in self.tabu_list_priority.items() if v > iteration}

            # 记录日志
            self.log_data.append({
                "Iteration": iteration + 1,
                "Best_Fitness_Current": current_fitness,
                "Best_Fitness_Global": self.best_fitness,
                "Tabu_List_Size": len(self.tabu_list_allocation) + len(self.tabu_list_priority),
                "Diversification_Applied": diversification_applied,
                "Best_Solution_Changed": solution_improved
            })

        # 记录最终结果
        self.log_data.append({
            "Iteration": self.max_iterations,
            "Best_Fitness_Current": "Final",
            "Best_Fitness_Global": self.best_fitness,
            "Tabu_List_Size": "N/A",
            "Diversification_Applied": "N/A",
            "Best_Solution_Changed": "N/A"
        })

        # 返回最优解
        return self.best_solution

    def save_logs_to_csv(self, filename="ts_logs.csv"):
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
    simulator.load_resource_info("../TTFM_data/computing_network_info.json")
    simulator.load_task_info("../TTFM_data/task_info.json")

    # 获取任务和节点数量
    num_nodes = len(simulator.resource_info["compute_power"])
    num_tasks = len(simulator.task_info["task_node_values"])

    # 运行禁忌搜索
    ts = TabuSearch(
        simulator=simulator,
        num_tasks=num_tasks,
        num_nodes=num_nodes,
        max_iterations=100,
        tabu_tenure=10,
        neighborhood_size=10,
        diversification_threshold=10
    )

    best_solution = ts.run()

    # 保存日志到CSV
    ts.save_logs_to_csv("output/ts_log.csv")

    # 打印日志数据
    print("Tabu Search Algorithm Logs:")
    for header in ts.log_headers:
        print(f"{header}", end=",")
    print()
    for row in ts.log_data:
        for header in ts.log_headers:
            print(f"{row[header]}", end=",")
        print()

    # 输出最优结果
    allocation, priority = best_solution
    allocation_matrix = [[0] * num_tasks for _ in range(num_nodes)]
    for task_id, node_id in enumerate(allocation):
        allocation_matrix[node_id][task_id] = 1

    print("\nBest Allocation Matrix:")
    for row in allocation_matrix:
        print(row)
    print("Best Priority Vector:", priority)