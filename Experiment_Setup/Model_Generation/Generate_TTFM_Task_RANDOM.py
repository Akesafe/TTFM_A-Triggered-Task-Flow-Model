import numpy as np
import json
import matplotlib.pyplot as plt
import networkx as nx
import random


# 保持不变，因为有外部脚本调用
def save_json_compact_lists(data, file_path, indent_level=1, line_width=80):
    def format_list(lst, line_width=80):
        """格式化列表，将最里面的列表显示为一行。"""
        indent = ""
        result = "["
        line = indent
        for i, item in enumerate(lst):
            item_str = json.dumps(item)  # 转换为字符串
            if len(line) + len(item_str) + 2 > line_width:
                result += line.rstrip() + "\n" + indent
                line = indent
            line += item_str + ", "
        result += line.rstrip(", ") + "]"
        return result

    def process_data(data, indent_level=0):
        """递归处理data，保证最里面的列表内容在一行显示。"""
        if isinstance(data, list):
            # 如果是列表且列表内不是复杂对象，压缩为一行显示
            if all(not isinstance(i, (list, dict)) for i in data):
                return format_list(data)
            else:
                # 否则逐个元素处理
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
            # 如果是字典，处理键值对
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
            # 其他类型直接返回
            return json.dumps(data)

    # 将处理后的内容保存到指定路径的JSON文件
    with open(file_path, 'w') as json_file:
        json_file.write(process_data(data, indent_level))


class TaskDAG:
    def __init__(self, num_of_task, random_seed=None):
        """
        初始化任务有向无环图生成器

        Args:
            num_of_task: 任务数量
            random_seed: 随机种子，用于保证结果可重现
        """
        self.num_of_task = num_of_task

        # 设置随机种子以保证可重现性
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        # 初始化所有属性为None
        self.adj_matrix = None
        self.task_node_values = None
        self.task_edge_values = None
        self.start_tasks = None
        self.end_tasks = None
        self.num_sources = None
        self.num_sinks = None
        self.sources_to_start_tasks = None
        self.end_tasks_to_sinks = None
        self.key_task_flows = None

    def generate_dag(self):
        """生成有向无环图并设置相关属性"""
        # 生成DAG的邻接矩阵、节点值和边值
        self.adj_matrix, self.task_node_values, self.task_edge_values = self._generate_random_dag()

        # 查找起始任务和末任务
        self.start_tasks, self.end_tasks = self._find_start_end_tasks()

        # 随机生成触发源和终点的数量
        self.num_sources = np.random.randint(1, len(self.start_tasks) + 1)  # 随机生成 ≤ 起始任务数量的触发源
        self.num_sinks = np.random.randint(1, len(self.end_tasks) + 1)  # 随机生成 ≤ 末任务数量的终点

        # 设置触发源和任务终点
        self.sources_to_start_tasks, self.end_tasks_to_sinks = self._set_sources_and_sinks()

        # 生成关键任务流
        self.key_task_flows = self._generate_key_task_flows()

        return self

    def _generate_random_dag(self):
        """
        随机生成有向无环图的邻接矩阵，节点值和边值

        Returns:
            adj_matrix: 邻接矩阵
            node_values: 节点值（计算量）
            edge_values: 边值（传输量）
        """

        def is_connected(adj_matrix):
            # 将邻接矩阵转换为无向图，以检查弱连通性
            G = nx.from_numpy_array(adj_matrix, create_using=nx.Graph)
            return nx.is_connected(G)

        while True:
            # 生成一个上三角矩阵，确保没有环
            random_matrix = np.random.uniform(0.01, 1, size=(self.num_of_task, self.num_of_task))  # 生成 n x n 的随机整数矩阵
            random_matrix = random_matrix * (
                        np.random.rand(self.num_of_task, self.num_of_task) > 0.6)  # 让每个元素有 60% 的概率变为 0
            adj_matrix = np.triu(random_matrix, 1)  # 提取上三角部分，不包含主对角线

            # 检查是否存在孤立节点
            if is_connected(adj_matrix):
                print("Task DAG generated")
                break  # 如果图是连通的，则退出循环

            print("There is isolated node in task DAG. regenerating...")

        # 生成随机节点值，代表计算量
        node_values = np.random.uniform(0.1, 10, size=self.num_of_task)

        # 边的值为邻接矩阵中的非零元素
        edge_values = adj_matrix[adj_matrix != 0]

        return adj_matrix, node_values, edge_values

    def _find_start_end_tasks(self):
        """
        确定起始任务和末任务

        Returns:
            start_tasks: 入度为0的任务
            end_tasks: 出度为0的任务
        """
        start_tasks = np.where(np.sum(self.adj_matrix, axis=0) == 0)[0]  # 入度为0的任务
        end_tasks = np.where(np.sum(self.adj_matrix, axis=1) == 0)[0]  # 出度为0的任务
        return start_tasks, end_tasks

    def _set_sources_and_sinks(self):
        """
        随机设置触发源和任务终点

        Returns:
            sources_to_start_tasks: 触发源到起始任务的连接矩阵
            end_tasks_to_sinks: 末任务到任务终点的连接矩阵
        """
        # 触发源到起始任务的矩阵：行数为任务总数，列数为触发源数
        sources_to_start_tasks = np.zeros((self.num_of_task, self.num_sources))

        # 遍历每个起始任务，随机分配一个触发源
        for start_task_idx in self.start_tasks:
            source_idx = np.random.randint(0, self.num_sources)  # 随机选择一个触发源
            sources_to_start_tasks[start_task_idx, source_idx] = np.random.uniform(0.001, 5)

        # 末任务到任务终点的矩阵：行数为任务总数，列数为终点数
        end_tasks_to_sinks = np.zeros((self.num_of_task, self.num_sinks))

        # 遍历每个末任务，随机分配一个任务终点
        for end_task_idx in self.end_tasks:
            sink_idx = np.random.randint(0, self.num_sinks)  # 随机选择一个任务终点
            end_tasks_to_sinks[end_task_idx, sink_idx] = np.random.uniform(0.001, 0.5)

        return sources_to_start_tasks, end_tasks_to_sinks

    def _generate_key_task_flows(self, num_key_flows=None):
        """
        生成关键任务流

        Args:
            num_key_flows: 需要生成的关键任务流数量

        Returns:
            key_flows: 关键任务流列表
        """
        key_flows = []
        if num_key_flows is None:
            # 根据任务数量x自适应调整，公式为2 * log10(x/2)四舍五入
            if self.num_of_task <= 2:  # Avoid log10(<=1) issues and ensure min
                calculated_num_flows = 2
            else:
                calculated_num_flows = int(round(2 * np.log10(self.num_of_task / 2)))
            # Apply constraints
            num_key_flows = max(2, min(calculated_num_flows, 6))
        while len(key_flows) < num_key_flows:  # 确保生成的关键任务流数量满足要求
            # 随机选择一个触发源
            source_idx = random.randint(0, self.sources_to_start_tasks.shape[1] - 1)
            source = f"Source {source_idx}"

            # 找到该触发源连接的任务
            start_tasks = [i for i, weight in enumerate(self.sources_to_start_tasks[:, source_idx]) if weight > 0]

            if not start_tasks:
                continue  # 如果没有关联的起始任务，跳过该源

            current_task = random.choice(start_tasks)  # 从触发源连接的任务中随机选择一个
            task_in_path = [current_task]  # 记录路径中的任务

            # 从当前任务开始，随机选择后继任务，直到到达某个终点
            while True:
                successors = [i for i, weight in enumerate(self.adj_matrix[current_task]) if weight > 0]
                sinks = [i for i, weight in enumerate(self.end_tasks_to_sinks[current_task]) if weight > 0]

                if successors and (random.random() > 0.15 or not sinks):
                    # 有后继任务且随机选择继续任务流（85%概率），或者没有终点可以选择
                    current_task = random.choice(successors)
                    task_in_path.append(current_task)
                elif sinks:
                    # 随机选择一个终点作为结束点
                    sink_idx = random.choice(sinks)
                    sink = f"Sink {sink_idx}"
                    break
                else:
                    # 没有后继任务也没有终点，结束生成
                    sink = None
                    break

            # 构造关键任务流
            if sink:
                key_flow = {
                    "source": source_idx,
                    "sink": int(sink.split(" ")[1]),
                    "task_in_path": task_in_path
                }

                # 检查新生成的关键任务流是否已存在
                if key_flow not in key_flows:
                    key_flows.append(key_flow)
                else:
                    print("There is repeated key task flow, skipped...")

        return key_flows

    def draw_dag(self):
        """绘制DAG图，显示任务、触发源和终点之间的关系"""
        # 创建有向图
        G = nx.DiGraph()

        # 添加任务节点及其属性（计算量）
        for i in range(len(self.task_node_values)):
            G.add_node(f"Task {i}", weight=self.task_node_values[i], node_type="task")

        # 添加任务之间的边及其权重
        for i in range(len(self.adj_matrix)):
            for j in range(len(self.adj_matrix[i])):
                if self.adj_matrix[i][j] != 0:
                    G.add_edge(f"Task {i}", f"Task {j}", weight=self.adj_matrix[i][j])

        # 添加触发源节点及其连接
        num_sources = self.sources_to_start_tasks.shape[1]
        for source_idx in range(num_sources):
            source_node = f"Source {source_idx}"
            G.add_node(source_node, node_type="source")  # 添加源节点
            for task_idx, weight in enumerate(self.sources_to_start_tasks[:, source_idx]):
                if weight > 0:
                    G.add_edge(source_node, f"Task {task_idx}", weight=weight)

        # 添加终点节点及其连接
        num_sinks = self.end_tasks_to_sinks.shape[1]
        for sink_idx in range(num_sinks):
            sink_node = f"Sink {sink_idx}"
            G.add_node(sink_node, node_type="sink")  # 添加终点节点
            for task_idx, weight in enumerate(self.end_tasks_to_sinks[:, sink_idx]):
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

    def get_output_data(self):
        """
        获取输出数据对象

        Returns:
            dict: 包含所有DAG信息的字典
        """
        data = {
            "adjacency_matrix": self.adj_matrix.tolist(),
            "task_node_values": self.task_node_values.tolist(),
            "task_edge_values": self.task_edge_values.tolist(),
            "sources_to_start_tasks": self.sources_to_start_tasks.tolist(),
            "end_tasks_to_sinks": self.end_tasks_to_sinks.tolist(),
            "start_tasks": self.start_tasks.tolist(),  # 保存起始任务列表
            "end_tasks": self.end_tasks.tolist(),  # 保存末任务列表
            "num_sources": self.num_sources,
            "num_sinks": self.num_sinks,
            "key_task_flows": self.key_task_flows
        }
        return data

    def save_output(self, file_path):
        """
        保存输出数据到JSON文件

        Args:
            file_path: 输出文件路径
        """
        data = self.get_output_data()
        save_json_compact_lists(data, file_path)
        print(f"Result has been saved in {file_path}")


if __name__ == "__main__":
    # 1. 初始化（设定节点数量，随机种子）
    NUM_of_TASK = 5  # 任务数
    task_dag = TaskDAG(num_of_task=NUM_of_TASK, random_seed=114515)

    # 2. 生成结果
    task_dag.generate_dag()
    result_file = "../../TTFM_data/task_info.json"
    task_dag.save_output(result_file)

    # 3. 显示图像（可选）
    task_dag.draw_dag()