import itertools
import csv

from TTFM_Simulation_main import TaskFlowSimulator


class TaskAllocationExhaustiveSearch:
    def __init__(self, simulator):
        self.simulator = simulator

    def generate_allocation_matrix(self, num_nodes, num_tasks):
        """生成所有可能的分配矩阵"""
        # 每个任务分配到某个节点，0 表示未分配，1 表示分配
        all_matrices = []
        node_task_combinations = itertools.product(range(num_nodes), repeat=num_tasks)
        for combo in node_task_combinations:
            matrix = [[0] * num_tasks for _ in range(num_nodes)]
            for task_id, node_id in enumerate(combo):
                matrix[node_id][task_id] = 1
            all_matrices.append(matrix)
        return all_matrices

    def generate_priority_vectors(self, num_tasks):
        """生成所有可能的任务优先级向量"""
        return list(itertools.permutations(range(1, num_tasks + 1)))

    def run_exhaustive_search(self, output_csv_path):
        """运行穷举搜索，输出结果到 CSV 文件"""
        num_nodes = len(self.simulator.resource_info["compute_power"])
        num_tasks = len(self.simulator.task_info["task_node_values"])

        # 生成所有可能的分配矩阵和优先级向量
        allocation_matrices = self.generate_allocation_matrix(num_nodes, num_tasks)
        priority_vectors = self.generate_priority_vectors(num_tasks)

        results = []

        # 穷举所有分配方案
        for allocation_matrix in allocation_matrices:
            for priority_vector in priority_vectors:
                # 直接设置 task_allocation_info，跳过格式检查
                self.simulator.task_allocation_info = {
                    "allocation_matrix": allocation_matrix,
                    "task_priorities": priority_vector
                }

                # 运行仿真
                latencies = self.simulator.simulate_key_task_flow_latency()
                for flow_id, latency_dict in latencies.items():
                    results.append({
                        "allocation_matrix": allocation_matrix,
                        "task_priorities": priority_vector,
                        "flow_id": flow_id,
                        "total_latency": latency_dict.get("total_latency"),
                        "total_package_latency": latency_dict.get("total_package_latency"),
                        "total_transfer_latency": latency_dict.get("total_transfer_latency"),
                        "total_compute_latency": latency_dict.get("total_compute_latency"),
                        "total_task_queueing_latency": latency_dict.get("total_task_queueing_latency")
                    })

        # 写入 CSV 文件
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = [
                "allocation_matrix",
                "task_priorities",
                "flow_id",
                "total_latency",
                "total_package_latency",
                "total_transfer_latency",
                "total_compute_latency",
                "total_task_queueing_latency"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                writer.writerow({
                    "allocation_matrix": result["allocation_matrix"],
                    "task_priorities": result["task_priorities"],
                    "flow_id": result["flow_id"],
                    "total_latency": result.get("total_latency"),
                    "total_package_latency": result.get("total_package_latency"),
                    "total_transfer_latency": result.get("total_transfer_latency"),
                    "total_compute_latency": result.get("total_compute_latency"),
                    "total_task_queueing_latency": result.get("total_task_queueing_latency")
                })
                break
        print(f"搜索完成，结果已保存到 {output_csv_path}")


# 示例使用
simulator = TaskFlowSimulator()
simulator.load_resource_info('TTFM_data/computing_network_info.json')
simulator.load_task_info('TTFM_data/task_info.json')

searcher = TaskAllocationExhaustiveSearch(simulator)
searcher.run_exhaustive_search("allocation_results.csv")
