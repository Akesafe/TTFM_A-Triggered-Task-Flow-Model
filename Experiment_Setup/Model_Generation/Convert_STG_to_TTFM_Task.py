import networkx as nx
import numpy as np
import json
import random
import os
import matplotlib.pyplot as plt

# Set a fixed seed for reproducibility
np.random.seed(114515)
random.seed(114515)


# 修改后的绘制DAG函数
def draw_dag(adj_matrix, node_values, sources_to_start_tasks, end_tasks_to_sinks):
    # 创建有向图
    G = nx.DiGraph()

    # 添加任务节点及其属性（计算量）
    for i in range(len(node_values)):
        G.add_node(f"Task {i}", weight=node_values[i], node_type="task")

    # 添加任务之间的边及其权重
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] != 0:
                G.add_edge(f"Task {i}", f"Task {j}", weight=adj_matrix[i][j])

    # 添加触发源节点及其连接
    num_sources = sources_to_start_tasks.shape[1]
    for source_idx in range(num_sources):
        source_node = f"Source {source_idx}"
        G.add_node(source_node, node_type="source")  # 添加源节点
        for task_idx, weight in enumerate(sources_to_start_tasks[:, source_idx]):
            if weight > 0:
                G.add_edge(source_node, f"Task {task_idx}", weight=weight)

    # 添加终点节点及其连接
    num_sinks = end_tasks_to_sinks.shape[1]
    for sink_idx in range(num_sinks):
        sink_node = f"Sink {sink_idx}"
        G.add_node(sink_node, node_type="sink")  # 添加终点节点
        for task_idx, weight in enumerate(end_tasks_to_sinks[:, sink_idx]):
            if weight > 0:
                G.add_edge(f"Task {task_idx}", sink_node, weight=weight)

    # 使用 circular_layout 布局
    pos = nx.circular_layout(G)

    # 定义节点颜色
    node_colors = []
    for node in G.nodes(data=True):
        if node[1]['node_type'] == 'task':
            node_colors.append('skyblue')
        elif node[1]['node_type'] == 'source':
            node_colors.append('lightgreen')
        elif node[1]['node_type'] == 'sink':
            node_colors.append('salmon')

    # 绘制节点
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=2000)

    # 显示节点标签（任务节点显示计算量，源和终点显示标签）
    labels = {}
    for node, data in G.nodes(data=True):
        if data["node_type"] == "task":
            labels[node] = f"{node}\n({data['weight']})"  # 显示任务节点和计算量
        else:
            labels[node] = node  # 仅显示源和终点标签
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12)

    # 绘制边和边的权重（传输数据量）
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)

    plt.title("Directed Acyclic Graph (DAG) with Sources, Tasks, and Sinks")
    plt.show()


def parse_stg_file(file_path):
    """Parse the STG model file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out comment lines and empty lines
    lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

    # Get number of real tasks (excluding virtual start/end)
    num_real_tasks = int(lines[0])

    tasks = []
    for i in range(1, len(lines)):
        parts = lines[i].split()
        if len(parts) < 3:  # Skip lines that don't have enough data
            continue
        task_id = int(parts[0])
        cost = float(parts[1])
        num_predecessors = int(parts[2])
        predecessors = [int(parts[3 + j]) for j in range(num_predecessors) if 3 + j < len(parts)]
        tasks.append({
            'id': task_id,
            'cost': cost,
            'predecessors': predecessors
        })

    return num_real_tasks, tasks


def create_adjacency_matrix(num_real_tasks, tasks):
    """Create the adjacency matrix from the tasks."""
    # Initialize adjacency matrix for real tasks only (1 to num_real_tasks)
    adj_matrix = np.zeros((num_real_tasks, num_real_tasks))

    # Fill the adjacency matrix based on predecessor relationships
    for task in tasks:
        if 1 <= task['id'] <= num_real_tasks:  # Only consider real tasks as destinations
            for pred in task['predecessors']:
                if 1 <= pred <= num_real_tasks:  # Only consider real tasks as sources
                    # If pred is a predecessor of task['id'], then there's an edge from pred to task['id']
                    # In our adjacency matrix, adj_matrix[i][j] means task i is a predecessor of task j
                    adj_matrix[pred - 1][task['id'] - 1] = np.random.uniform(0.01, 1)

    return adj_matrix


def extract_task_node_values(tasks, num_real_tasks):
    """Extract task node values (costs) from the tasks."""
    node_values = np.zeros(num_real_tasks)

    for task in tasks:
        if 1 <= task['id'] <= num_real_tasks:  # Only consider real tasks
            node_values[task['id'] - 1] = task['cost']

    return node_values


def find_start_end_tasks(tasks, num_real_tasks):
    """Find start and end tasks by analyzing the task graph."""
    # Start tasks are those with the virtual start task (0) as a predecessor
    start_tasks = []
    for task in tasks:
        if 1 <= task['id'] <= num_real_tasks and 0 in task['predecessors']:
            start_tasks.append(task['id'] - 1)  # Convert to 0-based index

    # End tasks are those that are predecessors of the virtual end task (num_real_tasks+1)
    end_tasks = []
    for task in tasks:
        if task['id'] == num_real_tasks + 1:  # Virtual end task
            end_tasks = [pred - 1 for pred in task['predecessors'] if 1 <= pred <= num_real_tasks]  # Convert to 0-based

    return start_tasks, end_tasks


def set_sources_and_sinks(num_real_tasks, start_tasks, end_tasks):
    """Set sources connected to start tasks and end tasks connected to sinks."""
    # Randomly decide the number of sources and sinks
    num_sources = np.random.randint(1, len(start_tasks) + 1)
    num_sinks = np.random.randint(1, len(end_tasks) + 1)

    # Initialize matrices
    sources_to_start_tasks = np.zeros((num_real_tasks, num_sources))
    end_tasks_to_sinks = np.zeros((num_real_tasks, num_sinks))

    # Connect sources to start tasks
    for start_task in start_tasks:
        source_idx = np.random.randint(0, num_sources)
        sources_to_start_tasks[start_task, source_idx] = np.random.uniform(0.001, 5)

    # Connect end tasks to sinks
    for end_task in end_tasks:
        sink_idx = np.random.randint(0, num_sinks)
        end_tasks_to_sinks[end_task, sink_idx] = np.random.uniform(0.001, 0.5)

    return sources_to_start_tasks, end_tasks_to_sinks, num_sources, num_sinks


def generate_key_task_flows(adj_matrix, sources_to_start_tasks, end_tasks_to_sinks, num_key_flows=4):
    """Generate key task flows from sources to sinks."""
    n = len(adj_matrix)
    key_flows = []

    attempts = 0
    max_attempts = 100  # Prevent infinite loops

    while len(key_flows) < num_key_flows and attempts < max_attempts:
        attempts += 1

        # Randomly select a source
        source_idx = random.randint(0, sources_to_start_tasks.shape[1] - 1)

        # Find tasks connected to this source
        start_tasks = [i for i, weight in enumerate(sources_to_start_tasks[:, source_idx]) if weight > 0]

        if not start_tasks:
            continue

        # Randomly select a start task
        current_task = random.choice(start_tasks)
        task_in_path = [current_task]

        # Start random walk through the graph until we reach a sink
        while True:
            # Find successors of current task
            # In the adj_matrix, j is a successor of i if adj_matrix[i][j] > 0
            successors = [j for j in range(n) if adj_matrix[current_task][j] > 0]

            # Check if the current task is connected to any sink
            sinks = [i for i, weight in enumerate(end_tasks_to_sinks[current_task]) if weight > 0]

            if successors and (random.random() > 0.3 or not sinks):
                # Continue to a successor
                current_task = random.choice(successors)
                task_in_path.append(current_task)
            elif sinks:
                # End the path at a sink
                sink_idx = random.choice(sinks)
                break
            else:
                # No way to continue or end, abort this path
                sink_idx = None
                break

        if sink_idx is not None:
            key_flow = {
                "source": source_idx,
                "sink": sink_idx,
                "task_in_path": task_in_path
            }

            # Avoid duplicate flows
            if key_flow not in key_flows:
                key_flows.append(key_flow)
                print(f"Generated key task flow {len(key_flows)}/{num_key_flows}")

    return key_flows


def save_json_compact_lists(data, file_path, indent_level=1, line_width=80):
    """Save data to a JSON file with compact formatting for lists."""

    def format_list(lst, line_width=80):
        """Format lists to be displayed on a single line."""
        indent = ""
        result = "["
        line = indent
        for i, item in enumerate(lst):
            item_str = json.dumps(item)
            if len(line) + len(item_str) + 2 > line_width:
                result += line.rstrip() + "\n" + indent
                line = indent
            line += item_str + ", "
        result += line.rstrip(", ") + "]"
        return result

    def process_data(data, indent_level=0):
        """Process data to format lists appropriately."""
        if isinstance(data, list):
            if all(not isinstance(i, (list, dict)) for i in data):
                return format_list(data)
            else:
                result = "[\n"
                for i, item in enumerate(data):
                    result += " " * ((indent_level + 1) * 2)
                    result += process_data(item, indent_level + 1)
                    if i < len(data) - 1:
                        result += ",\n"
                    else:
                        result += "\n"
                result += " " * (indent_level * 2) + "]"
                return result
        elif isinstance(data, dict):
            result = "{\n"
            for i, (key, value) in enumerate(data.items()):
                result += " " * ((indent_level + 1) * 2) + f'"{key}": '
                result += process_data(value, indent_level + 1)
                if i < len(data) - 1:
                    result += ",\n"
                else:
                    result += "\n"
            result += " " * (indent_level * 2) + "}"
            return result
        else:
            return json.dumps(data)

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as json_file:
        json_file.write(process_data(data, indent_level))


def convert_stg_to_json(stg_file_path, output_json_path):
    """Convert STG model to JSON format."""
    # Parse STG file
    print(f"Parsing STG file: {stg_file_path}")
    num_real_tasks, tasks = parse_stg_file(stg_file_path)
    print(f"Found {num_real_tasks} real tasks.")

    # Create adjacency matrix
    print("Creating adjacency matrix...")
    adj_matrix = create_adjacency_matrix(num_real_tasks, tasks)

    # Extract task node values
    print("Extracting task node values...")
    node_values = extract_task_node_values(tasks, num_real_tasks)

    # Extract edge values from the adjacency matrix
    edge_values = adj_matrix[adj_matrix != 0].tolist()

    # Find start and end tasks
    print("Finding start and end tasks...")
    start_tasks, end_tasks = find_start_end_tasks(tasks, num_real_tasks)
    print(f"Start tasks: {start_tasks}")
    print(f"End tasks: {end_tasks}")

    # Set sources and sinks
    print("Setting up sources and sinks...")
    sources_to_start_tasks, end_tasks_to_sinks, num_sources, num_sinks = set_sources_and_sinks(
        num_real_tasks, start_tasks, end_tasks)
    print(f"Number of sources: {num_sources}")
    print(f"Number of sinks: {num_sinks}")

    # Generate key task flows
    print("Generating key task flows...")
    key_task_flows = generate_key_task_flows(adj_matrix, sources_to_start_tasks, end_tasks_to_sinks)

    # Prepare data for JSON
    print("Preparing JSON data...")
    data = {
        "adjacency_matrix": adj_matrix.tolist(),
        "task_node_values": node_values.tolist(),
        "task_edge_values": edge_values,
        "sources_to_start_tasks": sources_to_start_tasks.tolist(),
        "end_tasks_to_sinks": end_tasks_to_sinks.tolist(),
        "start_tasks": start_tasks,
        "end_tasks": end_tasks,
        "num_sources": num_sources,
        "num_sinks": num_sinks,
        "key_task_flows": key_task_flows
    }

    # Save to JSON
    print(f"Saving result to {output_json_path}")
    save_json_compact_lists(data, output_json_path)
    print(f"Result has been saved in {output_json_path}")

    return data


# Sample main execution
if __name__ == "__main__":
    stg_file_path = "../STG_data/stg_model.stg"
    output_json_path = "../../TTFM_data/task_info.json"
    data = convert_stg_to_json(stg_file_path, output_json_path)

    # Draw the graph
    print("Drawing the graph visualization...")
    draw_dag(
        np.array(data["adjacency_matrix"]),
        np.array(data["task_node_values"]),
        np.array(data["sources_to_start_tasks"]),
        np.array(data["end_tasks_to_sinks"])
    )