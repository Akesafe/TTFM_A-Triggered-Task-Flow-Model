import json
import numpy as np
import random

# 使用自定义的save_json_compact_lists函数保存JSON文件
from Experiment_Setup.Model_Generation.Generate_TTFM_Task_RANDOM import save_json_compact_lists

# 使用固定的随机种子方便复现
random.seed(114514)

# Define file paths
task_info_path = '../TTFM_data/task_info.json'
resource_path = '../TTFM_data/computing_network_info.json'
output_file_path = '../TTFM_data/allocation_and_priority.json'

if __name__ == "__main__":
    # Load data from JSON files
    with open(task_info_path, 'r') as task_file:
        task_data = json.load(task_file)
        task_node_values = task_data["task_node_values"]

    with open(resource_path, 'r') as resource_file:
        resource_data = json.load(resource_file)
        compute_power = resource_data["compute_power"]

    # Determine number of tasks and compute nodes
    num_tasks = len(task_node_values)
    num_compute_nodes = len(compute_power)

    # Initialize the allocation matrix with zeros
    allocation_matrix = np.zeros((num_compute_nodes, num_tasks), dtype=int)

    # Assign each task to a compute_node randomly
    for task_idx in range(num_tasks):
        compute_idx = random.randint(0, num_compute_nodes - 1)
        allocation_matrix[compute_idx, task_idx] = 1

    # Generate a unique priority for each task
    task_priorities = random.sample(range(1, num_tasks + 1), num_tasks)

    # Convert the numpy array to a list for JSON serialization
    allocation_matrix_list = allocation_matrix.tolist()

    # Save the allocation matrix and priorities to JSON
    output_data = {
        "allocation_matrix": allocation_matrix_list,
        "task_priorities": task_priorities
    }
    save_json_compact_lists(output_data, output_file_path)
