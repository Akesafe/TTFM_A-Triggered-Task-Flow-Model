import random
import numpy as np
import csv
from TTFM_Simulation_main import TaskFlowSimulator

# 设定随机种子
RANDOM_SEED = 114514
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class Particle:
    """
    代表 PSO 群体中的一个粒子（一个潜在解）。
    """
    def __init__(self, num_tasks, num_nodes):
        """
        初始化粒子。

        属性:
        - position_allocation: 任务分配向量 (列表，索引为任务ID，值为节点ID)。
        - position_priority: 任务优先级向量 (列表，值为任务ID，顺序代表优先级)。
        - velocity_allocation: 分配速度。
        - velocity_priority: 优先级速度 (影响位置更新启发式)。
        - fitness: 适应度值 (平均延迟，越小越好)。
        - best_position_*: 个体历史最优位置。
        - best_fitness: 个体历史最优适应度。
        """
        # 任务分配：随机分配到节点
        self.position_allocation = [random.randint(0, num_nodes - 1) for _ in range(num_tasks)]
        # 任务优先级：随机排列 (1 到 num_tasks)
        self.position_priority = list(range(1, num_tasks + 1))
        random.shuffle(self.position_priority)

        # 速度：随机初始化
        self.velocity_allocation = [random.uniform(-1, 1) for _ in range(num_tasks)]
        self.velocity_priority = [random.uniform(-1, 1) for _ in range(num_tasks)]

        # 适应度
        self.fitness = float('inf')

        # 个体最优记录 (pbest)
        self.best_position_allocation = self.position_allocation[:]
        self.best_position_priority = self.position_priority[:]
        self.best_fitness = float('inf')


class ParticleSwarmOptimization:
    """
    实现粒子群优化算法，用于任务调度。
    """
    def __init__(self, simulator, num_tasks, num_nodes, num_particles=10, max_iterations=100, w=0.7, c1=1.5, c2=1.5):
        """
        初始化 PSO 算法。

        参数:
        - simulator: 任务流模拟器实例。
        - num_tasks: 任务数量。
        - num_nodes: 计算节点数量。
        - num_particles: 粒子数量。
        - max_iterations: 最大迭代次数。
        - w: 惯性权重。
        - c1: 个体学习因子 (pbest 影响)。
        - c2: 群体学习因子 (gbest 影响)。
        """
        self.simulator = simulator
        self.num_tasks = num_tasks
        self.num_nodes = num_nodes
        self.num_particles = num_particles
        self.max_iterations = max_iterations

        # PSO 参数
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # 初始化粒子群
        self.particles = [Particle(num_tasks, num_nodes) for _ in range(self.num_particles)]

        # 全局最优记录 (gbest)
        self.global_best_position_allocation = None
        self.global_best_position_priority = None
        self.global_best_fitness = float('inf')

        # 日志记录 (与 GA/SA/TS 风格一致)
        self.log_data = []
        self.log_headers = ["Iteration", "Average_Fitness_Current", "Best_Fitness_Current", "Best_Fitness_Global",
                            "Average_Velocity_Allocation", "Average_Velocity_Priority", "Best_Solution_Changed"]

    def evaluate_fitness(self, particle):
        """
        使用模拟器计算粒子的适应度（平均延迟）。
        """
        allocation_matrix = [[0] * self.num_tasks for _ in range(self.num_nodes)]
        # 确保节点 ID 在有效范围内
        for task_id, node_id in enumerate(particle.position_allocation):
            safe_node_id = int(max(0, min(node_id, self.num_nodes - 1))) # 保证分配在界内
            allocation_matrix[safe_node_id][task_id] = 1

        # 设置模拟器参数
        self.simulator.task_allocation_info = {
            "allocation_matrix": allocation_matrix,
            "task_priorities": particle.position_priority # 直接使用列表
        }

        # --- 调用模拟器，让错误自然抛出 ---
        latencies = self.simulator.simulate_key_task_flow_latency()
        # ---

        # 计算平均总时延
        total_latency = 0
        count = 0
        # 假设 latencies 是一个包含有效延迟信息的字典
        for flow_id in latencies:
             latency_value = latencies[flow_id].get("total_latency")
             # 简化：假设返回的 latency_value 是有效数字或 None
             if latency_value is not None:
                 total_latency += latency_value
                 count += 1
        # 如果无法计算（例如 count=0 或 latencies 为空），返回 inf
        return total_latency / count if count > 0 else float('inf')

    def update_velocity(self, particle):
        """
        根据惯性、pbest 和 gbest 更新粒子速度。
        """
        # 仅在 gbest 初始化后才更新速度
        if self.global_best_position_allocation is None:
             return

        for i in range(self.num_tasks):
            r1, r2 = random.random(), random.random()

            # 分配速度更新
            cognitive_alloc = self.c1 * r1 * (particle.best_position_allocation[i] - particle.position_allocation[i])
            social_alloc = self.c2 * r2 * (self.global_best_position_allocation[i] - particle.position_allocation[i])
            particle.velocity_allocation[i] = (self.w * particle.velocity_allocation[i] +
                                               cognitive_alloc + social_alloc)

            # 优先级速度更新 (其值主要影响下面位置更新中的交换概率)
            cognitive_prio = self.c1 * r1 * (particle.best_position_priority[i] - particle.position_priority[i])
            social_prio = self.c2 * r2 * (self.global_best_position_priority[i] - particle.position_priority[i])
            particle.velocity_priority[i] = (self.w * particle.velocity_priority[i] +
                                             cognitive_prio + social_prio)

    def update_position(self, particle):
        """
        根据速度更新粒子位置（分配和优先级）。
        """
         # 仅在 gbest 初始化后才更新位置
        if self.global_best_position_allocation is None:
            return

        new_allocation = particle.position_allocation[:]
        new_priority = particle.position_priority[:]

        # 分配位置更新 (概率性地移向 pbest 或 gbest)
        for i in range(self.num_tasks):
            # 使用速度绝对值计算移动概率 (类似 Sigmoid)
            prob_alloc = 1 / (1 + np.exp(-abs(particle.velocity_allocation[i])))
            if random.random() < prob_alloc:
                # 随机选择 pbest 或 gbest 作为目标
                target_node = particle.best_position_allocation[i] if random.random() < 0.5 else self.global_best_position_allocation[i]
                # 更新分配，确保节点ID有效
                new_allocation[i] = int(max(0, min(round(target_node), self.num_nodes - 1)))

        # 优先级位置更新 (基于速度的启发式交换)
        indices_to_consider = list(range(self.num_tasks))
        random.shuffle(indices_to_consider) # 打乱顺序避免偏见
        swapped_pairs = set() # 记录本轮已交换的对，避免重复

        for i in indices_to_consider:
            # 使用速度绝对值计算交换概率
            prob_prio_swap = 1 / (1 + np.exp(-abs(particle.velocity_priority[i])))
            if random.random() < prob_prio_swap:
                # 随机选择另一个不同的任务 j
                j = random.choice([idx for idx in range(self.num_tasks) if idx != i])
                pair = tuple(sorted((i, j))) # 创建无序对用于检查
                if pair not in swapped_pairs:
                    # 交换索引 i 和 j 处的优先级值
                    new_priority[i], new_priority[j] = new_priority[j], new_priority[i]
                    swapped_pairs.add(pair) # 标记为已交换

        # 应用更新
        particle.position_allocation = new_allocation
        particle.position_priority = new_priority
        # 注意：交换优先级值可以保持优先级列表是 1 到 N 的排列

    def run(self):
        """
        执行 PSO 优化过程。
        """
        # ---- 简化初始化 ----
        # 使用第一个粒子初始化 gbest (假设其 fitness 可计算，否则可能出错)
        first_particle = self.particles[0]
        first_particle.fitness = self.evaluate_fitness(first_particle)

        # 初始化第一个粒子的 pbest
        first_particle.best_fitness = first_particle.fitness
        first_particle.best_position_allocation = first_particle.position_allocation[:]
        first_particle.best_position_priority = first_particle.position_priority[:]

        # 初始化 gbest
        self.global_best_fitness = first_particle.fitness
        self.global_best_position_allocation = first_particle.position_allocation[:]
        self.global_best_position_priority = first_particle.position_priority[:]
        # ---- 初始化结束 ----

        # 主优化循环
        for iteration in range(self.max_iterations):
            sum_fitness = 0
            sum_velocity_allocation = 0
            sum_velocity_priority = 0
            solution_improved_in_iter = False

            for particle in self.particles:
                # 评估适应度 (每次迭代都评估)
                particle.fitness = self.evaluate_fitness(particle)
                sum_fitness += particle.fitness  # 直接累加，允许 inf

                # 更新个体最优 pbest
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position_allocation = particle.position_allocation[:]
                    particle.best_position_priority = particle.position_priority[:]

                # 更新全局最优 gbest
                if particle.fitness < self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position_allocation = particle.position_allocation[:]
                    self.global_best_position_priority = particle.position_priority[:]
                    solution_improved_in_iter = True

                # 累加绝对速度用于日志
                sum_velocity_allocation += np.mean(np.abs(particle.velocity_allocation))
                sum_velocity_priority += np.mean(np.abs(particle.velocity_priority))

            # 计算平均适应度和速度
            avg_fitness = sum_fitness / self.num_particles
            avg_velocity_allocation = sum_velocity_allocation / self.num_particles
            avg_velocity_priority = sum_velocity_priority / self.num_particles

            # 记录日志数据
            self.log_data.append({
                "Iteration": iteration + 1,
                "Average_Fitness_Current": avg_fitness, # 直接记录数值
                "Best_Fitness_Current": self.global_best_fitness, # 当前最优即为全局最优
                "Best_Fitness_Global": self.global_best_fitness, # 全局最优
                "Average_Velocity_Allocation": avg_velocity_allocation,
                "Average_Velocity_Priority": avg_velocity_priority,
                "Best_Solution_Changed": int(solution_improved_in_iter)
            })

            # 更新所有粒子的速度和位置
            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(particle)

        # 记录最终结果行 (与参考风格一致)
        final_avg_fitness = self.log_data[-1]["Average_Fitness_Current"] if self.log_data else "N/A"
        self.log_data.append({
            "Iteration": "Final",
            "Average_Fitness_Current": final_avg_fitness, # 使用最后一轮的平均值
            "Best_Fitness_Current": self.global_best_fitness, # 最终最优即全局最优
            "Best_Fitness_Global": self.global_best_fitness,
            "Average_Velocity_Allocation": "N/A",
            "Average_Velocity_Priority": "N/A",
            "Best_Solution_Changed": "N/A"
        })

        # 返回最优解
        return self.global_best_position_allocation, self.global_best_position_priority

    def save_logs_to_csv(self, filename="pso_logs.csv"):
        """
        将日志数据保存到 CSV 文件 (移除 try-except)。
        """
        # 假设 'output/' 目录存在或文件在当前目录
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.log_headers)
            writer.writeheader()
            writer.writerows(self.log_data)
        # print(f"日志已保存到 {filename}") # 可选的确认信息


# --- 主程序入口 (简化风格) ---
if __name__ == "__main__":
    # --- 数据文件路径 (直接使用相对路径) ---
    resource_file = '../TTFM_data/computing_network_info.json'
    task_info_file = '../TTFM_data/task_info.json'
    output_log_file = "output/pso_log.csv" # 输出日志文件名

    # --- 加载模拟器 (移除 try-except) ---
    simulator = TaskFlowSimulator()
    simulator.load_resource_info(resource_file)
    simulator.load_task_info(task_info_file)
    # print("模拟器加载完成.") # 可选信息

    # --- 获取任务和节点数量 ---
    num_nodes = len(simulator.resource_info["compute_power"])
    num_tasks = len(simulator.task_info["task_node_values"])

    # print(f"节点数量: {num_nodes}") # 可选信息
    # print(f"任务数量: {num_tasks}") # 可选信息

    # --- 配置并运行 PSO 算法 ---
    pso = ParticleSwarmOptimization(simulator, num_tasks, num_nodes,
                                    num_particles=20,      # 示例参数
                                    max_iterations=100,
                                    w=0.7, c1=1.5, c2=1.5)
    # print("开始运行 PSO...") # 可选信息
    best_allocation, best_priority = pso.run()
    # print("PSO 运行结束.") # 可选信息

    # --- 保存日志 ---
    pso.save_logs_to_csv(output_log_file)

    # --- 打印日志数据 (与参考风格一致) ---
    print("PSO 算法日志:")
    print(",".join(pso.log_headers)) # 打印表头
    for row in pso.log_data:
        # 直接按表头顺序打印值，假设 key 都存在
        print(",".join([str(row[h]) for h in pso.log_headers]))

    # --- 输出最优结果 (简化) ---
    print("\n最优分配矩阵:")
    allocation_matrix = [[0] * num_tasks for _ in range(num_nodes)]
    for task_id, node_id in enumerate(best_allocation):
        allocation_matrix[node_id][task_id] = 1
    for row in allocation_matrix:
        print(row)
    print("最优优先级向量:", best_priority)
    print(f"最优适应度 (平均延迟): {pso.global_best_fitness}")