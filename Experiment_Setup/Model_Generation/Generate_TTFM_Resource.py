import json
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 使用自定义的函数保存json文件
from Experiment_Setup.Model_Generation.Generate_TTFM_Task_RANDOM import save_json_compact_lists


class NetworkGenerator:
    """
    网络拓扑生成器，用于生成计算节点和通信链路
    """

    def __init__(self, num_of_compute_node=20, random_seed=114514):
        """
        初始化网络生成器

        Args:
            num_of_compute_node: 计算节点总数
            random_seed: 随机种子，方便复现
        """
        # ======================== 可调整参数 ========================
        # 计算节点总数
        self.NUM_OF_COMPUTE_NODE = num_of_compute_node

        # 节点类型比例（云:边:端）
        self.CLOUD_RATIO = 1
        self.EDGE_RATIO = 4
        self.END_RATIO = 16

        # 各类型节点的算力范围（单位：MIPS）
        self.CLOUD_POWER_RANGE = (5000, 10000)  # 云节点：高算力
        self.EDGE_POWER_RANGE = (1000, 5000)  # 边节点：中等算力
        self.END_POWER_RANGE = (100, 1000)  # 端节点：低算力

        # 连接概率参数
        self.EDGE_TO_CLOUD_PROB = 0.7  # 边节点连接云节点的概率
        self.EDGE_TO_EDGE_PROB = 0.5  # 边节点之间连接的概率
        self.END_TO_EDGE_PROB = 0.7  # 端节点连接边节点的概率

        # 延迟范围（单位：秒）
        self.CLOUD_EDGE_LATENCY_RANGE = (0.001, 0.01)  # 云节点到边节点的延迟
        self.EDGE_EDGE_LATENCY_RANGE = (0.0005, 0.005)  # 边节点之间的延迟
        self.EDGE_END_LATENCY_RANGE = (0.0002, 0.001)  # 边节点到端节点的延迟

        # 带宽范围（单位：Mbps）
        self.CLOUD_EDGE_BANDWIDTH_RANGE = (500, 1000)  # 云节点到边节点的带宽
        self.EDGE_EDGE_BANDWIDTH_RANGE = (100, 500)  # 边节点之间的带宽
        self.EDGE_END_BANDWIDTH_RANGE = (10, 100)  # 边节点到端节点的带宽

        # 设置随机种子
        self.RANDOM_SEED = random_seed
        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)

        # 数据存储
        self.node_types = []
        self.num_cloud = 0
        self.num_edge = 0
        self.num_end = 0
        self.compute_power = None
        self.latency_matrix = None
        self.bandwidth_matrix = None
        self.sources = []
        self.sinks = []
        self.source_to_node_latency = None
        self.source_to_node_bandwidth = None
        self.sink_to_node_latency = None
        self.sink_to_node_bandwidth = None
        self.trigger_types = []

    def categorize_nodes(self):
        """
        根据比例将节点分为云、边、端三种类型
        确保至少每种类型有一个节点（如果总节点数足够）
        """
        # 计算理想的节点数量
        total_ratio = self.CLOUD_RATIO + self.EDGE_RATIO + self.END_RATIO
        ideal_cloud = max(1, int(self.CLOUD_RATIO * self.NUM_OF_COMPUTE_NODE / total_ratio))
        ideal_edge = max(1, int(self.EDGE_RATIO * self.NUM_OF_COMPUTE_NODE / total_ratio))
        ideal_end = max(1, int(self.END_RATIO * self.NUM_OF_COMPUTE_NODE / total_ratio))

        # 调整节点数量，确保总数等于total_nodes
        while ideal_cloud + ideal_edge + ideal_end > self.NUM_OF_COMPUTE_NODE:
            # 优先减少数量较多的节点类型
            if ideal_end > ideal_edge and ideal_end > 1:
                ideal_end -= 1
            elif ideal_edge > ideal_cloud and ideal_edge > 1:
                ideal_edge -= 1
            elif ideal_cloud > 1:
                ideal_cloud -= 1
            else:
                # 如果所有类型都只有1个节点，则只能从端节点减少
                ideal_end -= 1

        while ideal_cloud + ideal_edge + ideal_end < self.NUM_OF_COMPUTE_NODE:
            # 按比例增加节点
            r = random.random() * total_ratio
            if r < self.CLOUD_RATIO:
                ideal_cloud += 1
            elif r < self.CLOUD_RATIO + self.EDGE_RATIO:
                ideal_edge += 1
            else:
                ideal_end += 1

        # 创建节点类型列表
        node_types = []
        node_types.extend(['cloud'] * ideal_cloud)
        node_types.extend(['edge'] * ideal_edge)
        node_types.extend(['end'] * ideal_end)

        # 打乱节点顺序
        random.shuffle(node_types)

        self.node_types = node_types
        self.num_cloud = ideal_cloud
        self.num_edge = ideal_edge
        self.num_end = ideal_end

        return node_types, ideal_cloud, ideal_edge, ideal_end

    def generate_compute_power(self):
        """
        根据节点类型分配算力
        云节点 > 边节点 > 端节点
        """
        compute_power = np.zeros(len(self.node_types))

        for i, node_type in enumerate(self.node_types):
            if node_type == 'cloud':
                compute_power[i] = random.uniform(*self.CLOUD_POWER_RANGE)
            elif node_type == 'edge':
                compute_power[i] = random.uniform(*self.EDGE_POWER_RANGE)
            else:  # end
                compute_power[i] = random.uniform(*self.END_POWER_RANGE)

        self.compute_power = compute_power
        return compute_power

    def generate_latency_bandwidth_matrices(self):
        """
        根据节点类型生成延迟和带宽矩阵
        遵循特定的连接规则
        """
        # 初始化矩阵
        latency_matrix = np.full((self.NUM_OF_COMPUTE_NODE, self.NUM_OF_COMPUTE_NODE), 999999.0)  # 大时延表示没有链路
        bandwidth_matrix = np.zeros((self.NUM_OF_COMPUTE_NODE, self.NUM_OF_COMPUTE_NODE))  # 带宽初始化为0

        # 设置节点与自身的时延为0，带宽为无限大
        np.fill_diagonal(latency_matrix, 0)
        np.fill_diagonal(bandwidth_matrix, 999999)

        # 获取每种类型的节点索引
        cloud_indices = [i for i, node_type in enumerate(self.node_types) if node_type == 'cloud']
        edge_indices = [i for i, node_type in enumerate(self.node_types) if node_type == 'edge']
        end_indices = [i for i, node_type in enumerate(self.node_types) if node_type == 'end']

        # 连接规则：
        # 1. 端节点只和边节点直接连接
        for end_idx in end_indices:
            # 确保每个端节点至少连接到一个边节点
            if not edge_indices:  # 如果没有边节点
                continue

            # 随机选择一个边节点连接
            edge_idx = random.choice(edge_indices)
            latency = random.uniform(*self.EDGE_END_LATENCY_RANGE)
            bandwidth = random.randint(*self.EDGE_END_BANDWIDTH_RANGE)

            latency_matrix[end_idx, edge_idx] = latency
            latency_matrix[edge_idx, end_idx] = latency  # 对称
            bandwidth_matrix[end_idx, edge_idx] = bandwidth
            bandwidth_matrix[edge_idx, end_idx] = bandwidth  # 对称

            # 根据概率连接更多边节点
            for edge_idx in edge_indices:
                if random.random() < self.END_TO_EDGE_PROB:
                    latency = random.uniform(*self.EDGE_END_LATENCY_RANGE)
                    bandwidth = random.randint(*self.EDGE_END_BANDWIDTH_RANGE)

                    latency_matrix[end_idx, edge_idx] = latency
                    latency_matrix[edge_idx, end_idx] = latency  # 对称
                    bandwidth_matrix[end_idx, edge_idx] = bandwidth
                    bandwidth_matrix[edge_idx, end_idx] = bandwidth  # 对称

        # 2. 边节点可以直接连接任何节点
        # 边节点和云节点的连接
        for edge_idx in edge_indices:
            for cloud_idx in cloud_indices:
                if random.random() < self.EDGE_TO_CLOUD_PROB:
                    latency = random.uniform(*self.CLOUD_EDGE_LATENCY_RANGE)
                    bandwidth = random.randint(*self.CLOUD_EDGE_BANDWIDTH_RANGE)

                    latency_matrix[edge_idx, cloud_idx] = latency
                    latency_matrix[cloud_idx, edge_idx] = latency  # 对称
                    bandwidth_matrix[edge_idx, cloud_idx] = bandwidth
                    bandwidth_matrix[cloud_idx, edge_idx] = bandwidth  # 对称

        # 边节点之间的连接
        for i, edge_i in enumerate(edge_indices):
            for edge_j in edge_indices[i + 1:]:  # 避免重复
                if random.random() < self.EDGE_TO_EDGE_PROB:
                    latency = random.uniform(*self.EDGE_EDGE_LATENCY_RANGE)
                    bandwidth = random.randint(*self.EDGE_EDGE_BANDWIDTH_RANGE)

                    latency_matrix[edge_i, edge_j] = latency
                    latency_matrix[edge_j, edge_i] = latency  # 对称
                    bandwidth_matrix[edge_i, edge_j] = bandwidth
                    bandwidth_matrix[edge_j, edge_i] = bandwidth  # 对称

        # 3. 云节点之间不直接连接（通过边节点转发）

        self.latency_matrix = latency_matrix
        self.bandwidth_matrix = bandwidth_matrix
        return latency_matrix, bandwidth_matrix

    def is_connected(self):
        """
        检查网络是否连通，使用深度优先搜索
        """
        if self.bandwidth_matrix is None:
            raise ValueError("需要先生成带宽矩阵")

        num_nodes = len(self.bandwidth_matrix)
        visited = [False] * num_nodes

        # 深度优先搜索(DFS)的递归实现
        def dfs(node):
            visited[node] = True
            for neighbor in range(num_nodes):
                # 检查邻居是否有连接，且是否已访问
                if self.bandwidth_matrix[node][neighbor] > 0 and not visited[neighbor]:
                    dfs(neighbor)

        # 从节点0开始遍历
        dfs(0)

        # 如果所有节点都已访问，则图是连通的
        return all(visited)

    def ensure_connectivity(self):
        """
        确保网络连通性，必要时添加额外的链路
        """
        if self.latency_matrix is None or self.bandwidth_matrix is None:
            raise ValueError("需要先生成延迟和带宽矩阵")

        num_nodes = len(self.node_types)
        G = nx.Graph()

        # 添加所有节点
        for i in range(num_nodes):
            G.add_node(i)

        # 添加所有已有的边
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if self.bandwidth_matrix[i, j] > 0:
                    G.add_edge(i, j)

        # 检查是否存在孤立的组件
        components = list(nx.connected_components(G))

        if len(components) > 1:
            print(f"发现 {len(components)} 个不相连的网络组件，正在添加链路以确保连通性...")
            # 获取每种类型的节点索引
            edge_indices = [i for i, node_type in enumerate(self.node_types) if node_type == 'edge']

            # 连接不同的组件
            while len(components) > 1:
                # 选择两个不同的组件
                comp1 = list(components[0])
                comp2 = list(components[1])

                # 找到可以连接的节点对（优先选择边节点）
                valid_pairs = []
                for i in comp1:
                    for j in comp2:
                        if (self.node_types[i] == 'edge' or self.node_types[j] == 'edge') and \
                                not ((self.node_types[i] == 'cloud' and self.node_types[j] == 'end') or
                                     (self.node_types[i] == 'end' and self.node_types[j] == 'cloud')):
                            valid_pairs.append((i, j))

                if valid_pairs:
                    i, j = random.choice(valid_pairs)
                else:
                    # 如果没有合适的边节点对，尝试其他连接方式
                    # 这里简单处理，选择一个边节点作为中介
                    if not edge_indices:
                        # 如果没有边节点，创建随机连接
                        i = random.choice(comp1)
                        j = random.choice(comp2)
                    else:
                        # 连接到边节点
                        edge_node = random.choice(edge_indices)
                        i = random.choice(comp1)
                        j = edge_node

                        # 如果已经连接，选择另一个
                        if self.bandwidth_matrix[i, j] > 0:
                            continue

                # 建立连接
                if (self.node_types[i] == 'edge' and self.node_types[j] == 'cloud') or \
                        (self.node_types[i] == 'cloud' and self.node_types[j] == 'edge'):
                    latency = random.uniform(*self.CLOUD_EDGE_LATENCY_RANGE)
                    bandwidth = random.randint(*self.CLOUD_EDGE_BANDWIDTH_RANGE)
                elif (self.node_types[i] == 'edge' and self.node_types[j] == 'edge'):
                    latency = random.uniform(*self.EDGE_EDGE_LATENCY_RANGE)
                    bandwidth = random.randint(*self.EDGE_EDGE_BANDWIDTH_RANGE)
                else:  # edge-end connection
                    latency = random.uniform(*self.EDGE_END_LATENCY_RANGE)
                    bandwidth = random.randint(*self.EDGE_END_BANDWIDTH_RANGE)

                self.latency_matrix[i, j] = latency
                self.latency_matrix[j, i] = latency  # 对称
                self.bandwidth_matrix[i, j] = bandwidth
                self.bandwidth_matrix[j, i] = bandwidth  # 对称

                # 更新图
                G.add_edge(i, j)
                components = list(nx.connected_components(G))

        return self.latency_matrix, self.bandwidth_matrix

    def connect_sources_sinks(self, num_sources, num_sinks):
        """
        连接源节点和目标节点，只连接到端节点
        """
        # 获取端节点索引
        end_indices = [i for i, node_type in enumerate(self.node_types) if node_type == 'end']

        if not end_indices:
            raise ValueError("没有端节点可供连接！至少需要一个端节点")

        # 随机选择源节点和终点连接的端节点
        # 使用sample如果需要唯一的连接，否则使用choices
        if num_sources <= len(end_indices):
            sources = random.sample(end_indices, num_sources)
        else:
            print(f"警告: 源节点数量({num_sources})大于端节点数量({len(end_indices)})，部分源节点将连接到相同的端节点")
            sources = random.choices(end_indices, k=num_sources)

        if num_sinks <= len(end_indices):
            sinks = random.sample(end_indices, num_sinks)
        else:
            print(f"警告: 终点数量({num_sinks})大于端节点数量({len(end_indices)})，部分终点将连接到相同的端节点")
            sinks = random.choices(end_indices, k=num_sinks)

        # 初始化四个矩阵
        source_to_node_latency = np.full((num_sources, self.NUM_OF_COMPUTE_NODE), 999999.0)
        source_to_node_bandwidth = np.zeros((num_sources, self.NUM_OF_COMPUTE_NODE))
        sink_to_node_latency = np.full((num_sinks, self.NUM_OF_COMPUTE_NODE), 999999.0)
        sink_to_node_bandwidth = np.zeros((num_sinks, self.NUM_OF_COMPUTE_NODE))

        # 为每个源节点连接到对应的端节点
        for i, source_connect_node in enumerate(sources):
            source_to_node_latency[i, source_connect_node] = random.uniform(0.0001, 0.0005)  # 更低的延迟
            source_to_node_bandwidth[i, source_connect_node] = random.randint(10, 100)  # 较低的带宽，适合端节点

        # 为每个终点连接到对应的端节点
        for i, sink_connect_node in enumerate(sinks):
            sink_to_node_latency[i, sink_connect_node] = random.uniform(0.0001, 0.0005)
            sink_to_node_bandwidth[i, sink_connect_node] = random.randint(10, 100)

        self.sources = sources
        self.sinks = sinks
        self.source_to_node_latency = source_to_node_latency
        self.source_to_node_bandwidth = source_to_node_bandwidth
        self.sink_to_node_latency = sink_to_node_latency
        self.sink_to_node_bandwidth = sink_to_node_bandwidth

        return sources, sinks, source_to_node_latency, source_to_node_bandwidth, sink_to_node_latency, sink_to_node_bandwidth

    def generate_trigger_type(self):
        """
        生成触发源的触发方式
        """
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

    def visualize_network(self):
        """
        可视化网络拓扑，包括计算节点、触发源和终点，并标识节点类型
        """
        if any(attr is None for attr in [self.compute_power, self.latency_matrix, self.bandwidth_matrix,
                                         self.source_to_node_latency, self.source_to_node_bandwidth,
                                         self.sink_to_node_latency, self.sink_to_node_bandwidth]):
            raise ValueError("需要先完成所有数据的生成")

        G = nx.Graph()  # 使用无向图，因为带宽和延迟矩阵是对称的

        # 添加计算节点
        num_compute_nodes = len(self.compute_power)
        for i in range(num_compute_nodes):
            G.add_node(f"Node {i}", node_type=self.node_types[i], weight=self.compute_power[i])

        # 添加计算节点之间的边
        for i in range(num_compute_nodes):
            for j in range(i + 1, num_compute_nodes):  # 只需添加上三角部分
                if self.bandwidth_matrix[i][j] > 0 and self.bandwidth_matrix[i][j] < 999999:  # 如果有带宽连接
                    G.add_edge(f"Node {i}", f"Node {j}",
                               bandwidth=self.bandwidth_matrix[i][j],
                               latency=self.latency_matrix[i][j])

        # 添加触发源节点及其连接
        for i, source in enumerate(self.sources):
            source_node = f"Source {i}"
            G.add_node(source_node, node_type="source")
            for j in range(num_compute_nodes):
                if self.source_to_node_bandwidth[i][j] > 0:
                    G.add_edge(source_node, f"Node {j}",
                               bandwidth=self.source_to_node_bandwidth[i][j],
                               latency=self.source_to_node_latency[i][j])

        # 添加终点节点及其连接
        for i, sink in enumerate(self.sinks):
            sink_node = f"Sink {i}"
            G.add_node(sink_node, node_type="sink")
            for j in range(num_compute_nodes):
                if self.sink_to_node_bandwidth[i][j] > 0:
                    G.add_edge(sink_node, f"Node {j}",
                               bandwidth=self.sink_to_node_bandwidth[i][j],
                               latency=self.sink_to_node_latency[i][j])

        pos = nx.circular_layout(G)

        # 定义节点颜色
        node_colors = []
        for node in G.nodes(data=True):
            if node[1]['node_type'] == 'cloud':
                node_colors.append('gold')
            elif node[1]['node_type'] == 'edge':
                node_colors.append('skyblue')
            elif node[1]['node_type'] == 'end':
                node_colors.append('lightgreen')
            elif node[1]['node_type'] == 'source':
                node_colors.append('lightpink')
            elif node[1]['node_type'] == 'sink':
                node_colors.append('salmon')

        plt.figure(figsize=(12, 8))

        # 绘制节点
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=1500)

        # 显示节点标签
        labels = {}
        for node, data in G.nodes(data=True):
            if data["node_type"] in ["cloud", "edge", "end"]:
                labels[node] = f"{node}\n({int(data['weight'])}, {data['node_type']})"
            else:
                labels[node] = node
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

        # 绘制边的权重（带宽和延迟）
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            edge_labels[(u, v)] = f"B:{data['bandwidth']}\nL:{data['latency']:.4f}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Network Topology with Sources, Compute Nodes, and Sinks")
        plt.show()

    def generate_network(self, task_info_path, output_path=None):
        """
        生成网络拓扑并保存结果

        Args:
            task_info_path: 任务信息配置文件路径
            output_path: 输出文件路径，如果为None则不保存文件

        Returns:
            资源数据字典
        """
        # 1. 将节点分为云、边、端三种类型
        self.categorize_nodes()
        print(f"节点分类: 云节点={self.num_cloud}, 边节点={self.num_edge}, 端节点={self.num_end}")

        # 2. 根据节点类型分配算力
        self.generate_compute_power()

        # 3. 生成延迟和带宽矩阵
        self.generate_latency_bandwidth_matrices()

        # 4. 确保网络连通
        self.ensure_connectivity()

        # 验证网络连通性
        if self.is_connected():
            print("所有节点间仍然连通")
        else:
            print("网络不连通，请检查代码")
            return None

        # 读取 task_info.json 文件
        with open(task_info_path, 'r') as f:
            task_info = json.load(f)

        # 获取触发源和终点的数量
        num_sources = task_info['num_sources']
        num_sinks = task_info['num_sinks']

        # 5. 连接源节点和终点（只连接到端节点）
        self.connect_sources_sinks(num_sources, num_sinks)

        # 生成触发源的触发方式
        self.trigger_types = [self.generate_trigger_type() for _ in range(num_sources)]

        # 将生成的资源数据保存到 computing_network_info.json 文件中
        resource_data = {
            "compute_power": self.compute_power.tolist(),
            "latency_matrix": self.latency_matrix.tolist(),
            "bandwidth_matrix": self.bandwidth_matrix.tolist(),
            "source_to_node_latency": self.source_to_node_latency.tolist(),
            "source_to_node_bandwidth": self.source_to_node_bandwidth.tolist(),
            "sink_to_node_latency": self.sink_to_node_latency.tolist(),
            "sink_to_node_bandwidth": self.sink_to_node_bandwidth.tolist(),
            "trigger_types": self.trigger_types,
            "node_types": self.node_types  # 添加节点类型信息
        }

        # 将数据保存到指定路径
        if output_path:
            save_json_compact_lists(resource_data, output_path)
            print(f"结果保存在{output_path}中")

        return resource_data


if __name__ == "__main__":
    # 1. 初始化（设定节点数量，随机种子）
    network_generator = NetworkGenerator(num_of_compute_node=50, random_seed=114517)

    # 2. 生成结果
    resource_data = network_generator.generate_network(
        task_info_path='../../TTFM_data/task_info.json',
        output_path='../../TTFM_data/computing_network_info.json'
    )

    # 可视化网络（可选）
    network_generator.visualize_network()