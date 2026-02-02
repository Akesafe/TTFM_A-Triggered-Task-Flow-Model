import json
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 使用自定义的save_json_compact_lists函数保存JSON文件
from Experiment_Setup.Model_Generation.Generate_TTFM_Task_RANDOM import save_json_compact_lists

# 使用固定的随机种子方便复现
random.seed(114514)

# 定义计算节点的数量
NUM_of_COMPUTE_NODE = 20


def generate_latency_bandwidth_matrices(num_of_compute_node):
    # 随机生成对称时延矩阵，单位为秒(s)
    latency_matrix = np.random.uniform(0.0002, 0.05, size=(num_of_compute_node, num_of_compute_node))

    # 随机生成对称带宽矩阵，单位为Mbps
    bandwidth_matrix = np.random.randint(10, 1001, size=(num_of_compute_node, num_of_compute_node))

    # 使矩阵对称
    # 保留上三角部分（包含对角线），其余部分置为 0
    latency_matrix = np.triu(latency_matrix)
    bandwidth_matrix = np.triu(bandwidth_matrix)

    # 将矩阵下三角部分设置为上三角的对称值
    latency_matrix = latency_matrix + latency_matrix.T - np.diag(np.diag(latency_matrix))
    bandwidth_matrix = bandwidth_matrix + bandwidth_matrix.T - np.diag(np.diag(bandwidth_matrix))

    # 设置 60% 的概率移除一些链路
    for i in range(num_of_compute_node):
        for j in range(i + 1, num_of_compute_node):  # 只遍历上三角矩阵
            if np.random.rand() < 0.6:  # 60% 概率移除链路
                latency_matrix[i, j] = 999999
                latency_matrix[j, i] = 999999
                bandwidth_matrix[i, j] = 0
                bandwidth_matrix[j, i] = 0

    # 设置节点与自身的时延为0，带宽为999999
    np.fill_diagonal(latency_matrix, 0)
    np.fill_diagonal(bandwidth_matrix, 999999)

    return latency_matrix, bandwidth_matrix


def is_connected(bandwidth_matrix):
    num_nodes = len(bandwidth_matrix)
    visited = [False] * num_nodes

    # 深度优先搜索(DFS)的递归实现
    def dfs(node):
        visited[node] = True
        for neighbor in range(num_nodes):
            # 检查邻居是否有连接，且是否已访问
            if bandwidth_matrix[node][neighbor] > 0 and not visited[neighbor]:
                dfs(neighbor)

    # 从节点0开始遍历
    dfs(0)

    # 如果所有节点都已访问，则图是连通的
    return all(visited)


def generate_trigger_type():
    trigger_type = random.choice(["Periodic", "Poisson", "Normal"])
    if trigger_type == "Periodic":
        return {"type": "Periodic", "T": random.uniform(0.004, 0.5)}
    elif trigger_type == "Poisson":
        return {"type": "Poisson", "lambda": random.uniform(0.1, 5.0)}
    elif trigger_type == "Normal":
        return {
            "type": "Normal",
            "mean": random.uniform(0.005, 1),
            "stddev": random.uniform(0.001, 0.05)
        }


def visualize_network(compute_power, latency_matrix, bandwidth_matrix,
                      source_to_node_latency, source_to_node_bandwidth,
                      sink_to_node_latency, sink_to_node_bandwidth,
                      sources, sinks):
    """
    可视化网络拓扑，包括计算节点、触发源和终点
    """
    G = nx.Graph()  # 使用无向图，因为带宽和延迟矩阵是对称的

    # 添加计算节点
    num_compute_nodes = len(compute_power)
    for i in range(num_compute_nodes):
        G.add_node(f"Node {i}", node_type="compute", weight=compute_power[i])

    # 添加计算节点之间的边
    for i in range(num_compute_nodes):
        for j in range(i + 1, num_compute_nodes):  # 只需添加上三角部分
            if bandwidth_matrix[i][j] > 0:  # 如果有带宽连接
                G.add_edge(f"Node {i}", f"Node {j}",
                           bandwidth=bandwidth_matrix[i][j],
                           latency=latency_matrix[i][j])

    # 添加触发源节点及其连接
    for i, source in enumerate(sources):
        source_node = f"Source {i}"
        G.add_node(source_node, node_type="source")
        for j in range(num_compute_nodes):
            if source_to_node_bandwidth[i][j] > 0:
                G.add_edge(source_node, f"Node {j}",
                           bandwidth=source_to_node_bandwidth[i][j],
                           latency=source_to_node_latency[i][j])

    # 添加终点节点及其连接
    for i, sink in enumerate(sinks):
        sink_node = f"Sink {i}"
        G.add_node(sink_node, node_type="sink")
        for j in range(num_compute_nodes):
            if sink_to_node_bandwidth[i][j] > 0:
                G.add_edge(sink_node, f"Node {j}",
                           bandwidth=sink_to_node_bandwidth[i][j],
                           latency=sink_to_node_latency[i][j])

    pos = nx.circular_layout(G)

    # 定义节点颜色
    node_colors = []
    for node in G.nodes(data=True):
        if node[1]['node_type'] == 'compute':
            node_colors.append('skyblue')
        elif node[1]['node_type'] == 'source':
            node_colors.append('lightgreen')
        elif node[1]['node_type'] == 'sink':
            node_colors.append('salmon')

    plt.figure(figsize=(12, 8))

    # 绘制节点
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=1500)

    # 显示节点标签
    labels = {}
    for node, data in G.nodes(data=True):
        if data["node_type"] == "compute":
            labels[node] = f"{node}\n({data['weight']})"
        else:
            labels[node] = node
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # 绘制边的权重（带宽和延迟）
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_labels[(u, v)] = f"B:{data['bandwidth']}\nL:{data['latency']}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Network Topology with Sources, Compute Nodes, and Sinks")
    plt.show()

if __name__ == "__main__":
    # 随机生成算力值向量，假设算力值在 100 到 10000 之间 (单位：MIPS）
    compute_power = np.random.uniform(100, 10001, size=NUM_of_COMPUTE_NODE)

    # 确保生成连通的矩阵
    while True:
        latency_matrix, bandwidth_matrix = generate_latency_bandwidth_matrices(NUM_of_COMPUTE_NODE)
        if is_connected(bandwidth_matrix):
            print("所有节点间仍然连通")
            break
        else:
            print("存在孤立节点或无法到达的节点，重新生成矩阵")

    # 读取 task_info.json 文件
    with open('../../TTFM_data/task_info.json', 'r') as f:
        task_info = json.load(f)

    # 获取触发源和终点的数量
    num_sources = task_info['num_sources']
    num_sinks = task_info['num_sinks']

    # 随机选择源节点和终点
    sources = random.sample(range(NUM_of_COMPUTE_NODE), num_sources)
    sinks = random.sample(range(NUM_of_COMPUTE_NODE), num_sinks)

    # 初始化四个矩阵，源节点到各节点的时延和带宽，终点到各节点的时延和带宽
    source_to_node_latency = np.full((num_sources, NUM_of_COMPUTE_NODE), 999999.0)  # 大时延表示没有链路
    source_to_node_bandwidth = np.zeros((num_sources, NUM_of_COMPUTE_NODE))  # 带宽初始化为0

    sink_to_node_latency = np.full((num_sinks, NUM_of_COMPUTE_NODE), 999999.0)  # 大时延表示没有链路
    sink_to_node_bandwidth = np.zeros((num_sinks, NUM_of_COMPUTE_NODE))  # 带宽初始化为0

    # 为每个源节点和终点随机分配链路连通的节点
    for i, source in enumerate(sources):
        connected_node = random.choice(range(NUM_of_COMPUTE_NODE))  # 随机选择一个节点连接
        source_to_node_latency[i, connected_node] = np.random.uniform(0.0002, 0.001)  # 随机生成200us到1ms之间的时延
        source_to_node_bandwidth[i, connected_node] = np.random.randint(10, 1001)  # 随机生成10到1000之间的带宽

    for i, sink in enumerate(sinks):
        connected_node = random.choice(range(NUM_of_COMPUTE_NODE))  # 随机选择一个节点连接
        sink_to_node_latency[i, connected_node] = np.random.uniform(0.0002, 0.001)  # 随机生成200us到1ms之间的时延
        sink_to_node_bandwidth[i, connected_node] = np.random.randint(10, 1001)  # 随机生成10到1000之间的带宽

    # 生成触发源的触发方式
    trigger_types = [generate_trigger_type() for _ in range(num_sources)]

    # 将生成的资源数据保存到 computing_network_info.json 文件中
    resource_data = {
        "compute_power": compute_power.tolist(),
        "latency_matrix": latency_matrix.tolist(),
        "bandwidth_matrix": bandwidth_matrix.tolist(),
        "source_to_node_latency": source_to_node_latency.tolist(),
        "source_to_node_bandwidth": source_to_node_bandwidth.tolist(),
        "sink_to_node_latency": sink_to_node_latency.tolist(),
        "sink_to_node_bandwidth": sink_to_node_bandwidth.tolist(),
        "trigger_types": trigger_types
    }

    # 将数据保存到指定路径
    output_file = '../../TTFM_data/computing_network_info.json'
    save_json_compact_lists(resource_data, output_file)
    print(f"结果保存在{output_file}中")

    visualize_network(
        compute_power=compute_power,
        latency_matrix=latency_matrix,
        bandwidth_matrix=bandwidth_matrix,
        source_to_node_latency=source_to_node_latency,
        source_to_node_bandwidth=source_to_node_bandwidth,
        sink_to_node_latency=sink_to_node_latency,
        sink_to_node_bandwidth=sink_to_node_bandwidth,
        sources=sources,
        sinks=sinks
    )
