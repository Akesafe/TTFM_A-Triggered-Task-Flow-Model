import random
import itertools
import csv
from TTFM_Simulation_main import TaskFlowSimulator

random.seed(1)


class GeneticAlgorithm:
    def __init__(self, simulator, num_tasks, num_nodes, population_size=10, max_generations=100, crossover_rate=0.8,
                 mutation_rate=0.2):
        """
        初始化遗传算法的参数。
        """
        self.simulator = simulator
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        # 日志记录
        self.log_data = []
        self.log_headers = ["Iteration", "Average_Fitness_Current", "Best_Fitness_Current", "Best_Fitness_Global",
                            "Population_Diversity", "Best_Solution_Changed"]

    def initialize_population(self):
        """
        初始化种群，每个个体包含任务分配和优先级向量。
        """
        population = []
        for _ in range(self.population_size):
            # 随机分配任务到节点
            allocation = [random.randint(0, self.num_nodes - 1) for _ in range(self.num_tasks)]
            # 随机生成任务优先级向量
            priority = list(range(1, self.num_tasks + 1))
            random.shuffle(priority)
            population.append((allocation, priority))
        return population

    def evaluate_fitness(self, individual):
        """
        评估个体的适应度（目标是最小化关键任务流的平均总时延）。
        """
        allocation_matrix = [[0] * self.num_tasks for _ in range(self.num_nodes)]
        for task_id, node_id in enumerate(individual[0]):
            allocation_matrix[node_id][task_id] = 1

        # 设置任务分配和优先级
        self.simulator.task_allocation_info = {
            "allocation_matrix": allocation_matrix,
            "task_priorities": individual[1]
        }

        # 获取延迟信息
        latencies = self.simulator.simulate_key_task_flow_latency()

        # 计算平均总时延
        total_latency = 0
        count = 0
        for flow_id, latency_dict in latencies.items():
            total_latency += latency_dict.get("total_latency", 0)
            count += 1

        return total_latency / count if count > 0 else float('inf')  # 防止除零

    def selection(self, population, fitness_scores):
        """
        使用轮盘赌选择算法，从种群中选择适应度较高（fitness_scores较低）的个体。
        因为fitness_scores的值是时延值，根据优化目标，它的值越小越好。
        """
        # total_fitness = sum(fitness_scores)
        # probabilities = [1 - (f / total_fitness) for f in fitness_scores]  # 适应度越小，概率越高
        probabilities = [1/f for f in fitness_scores]
        selected_index = random.choices(range(len(population)), weights=probabilities, k=1)[0]
        return population[selected_index]

    def crossover(self, parent1, parent2):
        """
        交叉操作：对任务分配和优先级分别进行交叉。
        """
        # 任务分配的交叉
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.num_tasks - 1)
            child_allocation = parent1[0][:point] + parent2[0][point:]
        else:
            child_allocation = parent1[0].copy()

        # 优先级向量的交叉（次序交叉）
        if random.random() < self.crossover_rate:
            point1, point2 = sorted(random.sample(range(self.num_tasks), 2))
            sub_priority = parent1[1][point1:point2]
            remaining = [task for task in parent2[1] if task not in sub_priority]
            child_priority = remaining[:point1] + sub_priority + remaining[point1:]
        else:
            child_priority = parent1[1].copy()

        return (child_allocation, child_priority)

    def mutate(self, individual):
        """
        变异操作：随机调整任务分配或优先级。
        """
        # 任务分配的变异
        if random.random() < self.mutation_rate:
            task_id = random.randint(0, self.num_tasks - 1)
            new_node = random.randint(0, self.num_nodes - 1)
            individual[0][task_id] = new_node

        # 优先级的变异
        if random.random() < self.mutation_rate:
            task1, task2 = random.sample(range(self.num_tasks), 2)
            individual[1][task1], individual[1][task2] = individual[1][task2], individual[1][task1]

        return individual

    def calculate_diversity(self, population):
        """
        计算种群多样性指标
        """
        # 简单使用分配模式的种类数量作为多样性指标
        allocation_patterns = set()
        for ind in population:
            allocation_patterns.add(tuple(ind[0]))
        return len(allocation_patterns) / self.population_size  # 归一化到0-1范围

    def run(self):
        """
        执行遗传算法，返回最优的任务分配和优先级方案。
        """
        # 初始化种群
        population = self.initialize_population()

        # 记录最优解
        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.max_generations):
            # 评估适应度
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]

            # 计算当前代的平均适应度
            avg_fitness = sum(fitness_scores) / len(fitness_scores)

            # 找出当前代的最优个体
            current_best_index = fitness_scores.index(min(fitness_scores))
            current_best_fitness = fitness_scores[current_best_index]
            current_best_individual = population[current_best_index]

            # 计算种群多样性
            diversity = self.calculate_diversity(population)

            # 检查是否找到新的全局最优解
            solution_improved = False
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best_individual
                solution_improved = True

            # 记录日志
            self.log_data.append({
                "Iteration": generation + 1,
                "Average_Fitness_Current": avg_fitness,
                "Best_Fitness_Current": current_best_fitness,
                "Best_Fitness_Global": best_fitness,
                "Population_Diversity": diversity,
                "Best_Solution_Changed": int(solution_improved)
            })

            new_population = []

            # 选择下一代
            # 保留当前代的最优个体
            elite = population[current_best_index]
            # 生成其余新个体
            # new_population = [elite]  # 加入精英
            # 备注，出于测试需要，这里暂时不启用保留精英的功能。
            for _ in range(self.population_size-1+1):  # 减少1个位置给精英
                parent1 = self.selection(population, fitness_scores)
                parent2 = self.selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

        # 记录最终结果
        self.log_data.append({
            "Iteration": self.max_generations,
            "Average_Fitness_Current": "Final",
            "Best_Fitness_Current": best_fitness,
            "Best_Fitness_Global": best_fitness,
            "Population_Diversity": "N/A",
            "Best_Solution_Changed": "N/A"
        })

        return best_individual

    def save_logs_to_csv(self, filename="ga_logs.csv"):
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

    # 运行遗传算法
    ga = GeneticAlgorithm(
        simulator, num_tasks, num_nodes,
        population_size=10,
        max_generations=100,
        crossover_rate=0.1,
        mutation_rate=0.1
    )
    best_solution = ga.run()

    # 输出最优结果
    allocation_matrix = [[0] * num_tasks for _ in range(num_nodes)]
    for task_id, node_id in enumerate(best_solution[0]):
        allocation_matrix[node_id][task_id] = 1

    # 保存日志到CSV
    ga.save_logs_to_csv("output/ga_log.csv")

    # 打印日志数据
    print("GA Algorithm Logs:")
    for header in ga.log_headers:
        print(f"{header}", end=",")
    print()
    for row in ga.log_data:
        for header in ga.log_headers:
            print(f"{row[header]}", end=",")
        print()

    # 打印最终结果
    print("\nBest Allocation Matrix:")
    for row in allocation_matrix:
        print(row)
    print("Best Priority Vector:", best_solution[1])