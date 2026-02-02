import json

"""
---TTFM_data/task_info.json---
{
"adjacency_matrix": [
  [0, 0, 7, 0, 7, 0],
  [0, 0, 8, 7, 0, 6],
  [0, 0, 0, 0, 3, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0]
],
"task_node_values": [53, 87, 34, 64, 87, 99],
"task_edge_values": [7, 7, 8, 7, 6, 3],
"sources_to_start_tasks": [
  [8.0, 0.0],
  [0.0, 3.0],
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.0],
  [0.0, 0.0]
],
"end_tasks_to_sinks": [
  [0.0],
  [0.0],
  [0.0],
  [9.0],
  [3.0],
  [8.0]
],
"start_tasks": [0, 1],
"end_tasks": [3, 4, 5],
"num_sources": 2,
"num_sinks": 1
"key_task_flows": [
  {
    "source": 0,
    "sink": 0,
    "task_in_path": [0, 2, 4]
  }
}
"""
"""
---TTFM_data/computing_network_info.json---
{
    "compute_power": [84, 12, 23, 22],
    "latency_matrix": [
      [0, 34, 54, 72],
      [34, 0, 999999, 999999],
      [54, 999999, 0, 999999],
      [72, 999999, 999999, 0]
    ],
    "bandwidth_matrix": [
      [999999, 193, 318, 501],
      [193, 999999, 0, 0],
      [318, 0, 999999, 0],
      [501, 0, 0, 999999]
    ],
    "source_to_node_latency": [
      [999999, 50, 999999, 999999],
      [999999, 999999, 66, 999999]
    ],
    "source_to_node_bandwidth": [
      [0.0, 424.0, 0.0, 0.0],
      [0.0, 0.0, 242.0, 0.0]
    ],
    "sink_to_node_latency": [
      [999999, 999999, 30, 999999]
    ],
    "sink_to_node_bandwidth": [
      [0.0, 0.0, 67.0, 0.0]
    ],
    "trigger_types": [
      {
        "type": "Normal",
        "mean": 14.469829635181117,
        "stddev": 9.185110816308708
      },
      {
        "type": "Periodic",
        "T": 2
      }
    ]
  }
# 如果trigger_type为Normal（正态分布），触发间隔时间必须确保是正数
# 除了上面的例子，trigger_type还有一种可能的类型是Poisson，包含参数lambda，代表泊松分布
"""
"""
---TTFM_data/allocation_and_priority.json---
{
  "allocation_matrix": [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0]
  ],
  "task_priorities": [5, 2, 4, 3, 1, 6]
}
"""
"""
利用脚本ask_ChatGPT.py可以生成ask_ChatGPT_content.txt文件，把这个文件发给AI可以获取代码的详细解读。您也可以根据需要调整脚本来理解特定的代码片段或论文内容
"""


class TaskFlowSimulator:
    def __init__(self):
        """初始化任务流仿真器"""
        self.task_info = None
        self.resource_info = None
        self.task_allocation_info = None
        self.task_trigger_frequency = None

    def load_resource_info(self, resource_info_path):
        """加载资源信息并进行预处理"""
        self.resource_info = self._load_json(resource_info_path)
        print("resource loaded")
        if self.task_info is not None:
            self._process_resource_and_task_info()
            print("pre-process finished")

    def load_task_info(self, task_info_path):
        """加载任务信息"""
        self.task_info = self._load_json(task_info_path)
        print("task_info loaded")
        if self.resource_info is not None:
            self._process_resource_and_task_info()
            print("pre-process finished")

    def load_allocation_info(self, allocation_info_path):
        """加载任务分配信息"""
        self.task_allocation_info = self._load_json(allocation_info_path)
        # 检查分配矩阵的尺寸是否与任务数量和节点数量匹配
        num_tasks = len(self.task_info["task_node_values"])
        num_nodes = len(self.resource_info["compute_power"])
        allocation_matrix = self.task_allocation_info.get("allocation_matrix", [])
        if len(allocation_matrix) != num_nodes:
            raise ValueError(f"分配矩阵的行数（{len(allocation_matrix)}）与节点数量（{num_nodes}）不匹配")
        if any(len(row) != num_tasks for row in allocation_matrix):
            raise ValueError(f"分配矩阵的列数与任务数量（{num_tasks}）不匹配")
        # 检查优先级向量长度是否与任务数量一致，且是否各不相等
        task_priorities = self.task_allocation_info.get("task_priorities", [])
        if len(task_priorities) != num_tasks:
            raise ValueError(f"任务优先级向量长度（{len(task_priorities)}）与任务数量（{num_tasks}）不一致")
        if len(set(task_priorities)) != len(task_priorities):
            raise ValueError("任务优先级向量中的值不唯一，存在重复")
        print("allocation_info loaded")

    @staticmethod
    def _load_json(file_path):
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _process_resource_and_task_info(self):
        """执行格式检查"""
        # 检查任务信息中的触发源数量和终点数量是否定义
        if 'num_sources' not in self.task_info or 'num_sinks' not in self.task_info:
            raise ValueError("任务信息中缺少 'num_sources' 或 'num_sinks' 字段")

        # 检查触发源数量和终点数量是否对应
        num_sources_in_task = self.task_info.get("num_sources", 0)
        num_sinks_in_task = self.task_info.get("num_sinks", 0)
        num_sources_in_resource = len(self.resource_info.get("source_to_node_latency", []))
        num_sinks_in_resource = len(self.resource_info.get("sink_to_node_latency", []))
        if num_sources_in_task != num_sources_in_resource:
            raise ValueError(f"触发源数量不一致：任务信息中为 {num_sources_in_task}，资源信息中为 {num_sources_in_resource}")
        if num_sinks_in_task != num_sinks_in_resource:
            raise ValueError(f"终点数量不一致：任务信息中为 {num_sinks_in_task}，资源信息中为 {num_sinks_in_resource}")

        """计算等效时延和带宽，更新资源信息"""
        self.resource_info['latency_matrix'], self.resource_info[
            'bandwidth_matrix'] = self.calculate_equivalent_bandwidth(
            self.resource_info['latency_matrix'], self.resource_info['bandwidth_matrix']
        )
        self.resource_info['source_to_node_latency'], self.resource_info[
            'source_to_node_bandwidth'] = self.calculate_source_to_node_metrics(
            self.resource_info['latency_matrix'],
            self.resource_info['source_to_node_latency'],
            self.resource_info['bandwidth_matrix'],
            self.resource_info['source_to_node_bandwidth']
        )
        self.resource_info['sink_to_node_latency'], self.resource_info[
            'sink_to_node_bandwidth'] = self.calculate_sink_to_node_metrics(
            self.resource_info['latency_matrix'],
            self.resource_info['sink_to_node_latency'],
            self.resource_info['bandwidth_matrix'],
            self.resource_info['sink_to_node_bandwidth']
        )
        self.task_trigger_frequency = self.calculate_task_trigger_frequency()

        num_tasks = len(self.task_info["task_node_values"])
        num_nodes = len(self.resource_info["compute_power"])
        print(f"num_tasks:{num_tasks}; num_nodes:{num_nodes}")
        print(f"source_num:{num_sources_in_task}; num_sinks:{num_sinks_in_task}")

    def calculate_equivalent_bandwidth(self, latency_matrix, bandwidth_matrix):
        """计算所有节点间的等效总时延和等效总带宽"""
        '''
        生成完整的node-to-node的时延和带宽
          - 不同路径上的时延和带宽值可能不一样
          - 考虑到实际场景中很少会出现这样的情况：
            - “两个节点之间有两条链路
            - 其中一条是大带宽高时延
            - 另一条是小带宽低时延”
          - 因此统一采用时延最低的那条链路计算带宽和时延，算法如下：
            - 等效总时延=min（所有链路时延之和）
            - 等效总带宽=min（计算出的等效总时延所在链路上各链路的带宽）
        '''

        def floyd_warshall(matrix):
            n = len(matrix)
            distance = [row[:] for row in matrix]
            next_node = [[None if matrix[i][j] == 999999 else j for j in range(n)] for i in range(n)]
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if distance[i][j] > distance[i][k] + distance[k][j]:
                            distance[i][j] = distance[i][k] + distance[k][j]
                            next_node[i][j] = next_node[i][k]
            return distance, next_node

        def find_path(i, j, next_node):
            path = []
            while i is not None:
                path.append(i)
                if i == j:
                    break
                i = next_node[i][j]
            return path

        min_latency_matrix, next_node = floyd_warshall(latency_matrix)
        n = len(latency_matrix)
        equivalent_bandwidth = [[0 if i != j else 999999 for j in range(n)] for i in range(n)]

        for i in range(n):
            for j in range(n):
                if i != j:
                    path = find_path(i, j, next_node)
                    min_bandwidth = min(bandwidth_matrix[path[k]][path[k + 1]] for k in range(len(path) - 1))
                    equivalent_bandwidth[i][j] = min_bandwidth

        return min_latency_matrix, equivalent_bandwidth

    def calculate_source_to_node_metrics(self, latency_matrix, source_to_node_latency, bandwidth_matrix,
                                         source_to_node_bandwidth):
        """计算source到node的等效时延和带宽"""
        num_sources = len(source_to_node_latency)
        num_nodes = len(latency_matrix)
        equivalent_latency = [[999999] * num_nodes for _ in range(num_sources)]
        equivalent_bandwidth = [[0] * num_nodes for _ in range(num_sources)]

        for s in range(num_sources):
            for target in range(num_nodes):
                min_latency = 999999
                corresponding_bandwidth = 0
                direct_latency = source_to_node_latency[s][target]
                if direct_latency != 999999:
                    min_latency = direct_latency
                    corresponding_bandwidth = source_to_node_bandwidth[s][target]
                for intermediate in range(num_nodes):
                    if source_to_node_latency[s][intermediate] != 999999:
                        total_latency = source_to_node_latency[s][intermediate] + latency_matrix[intermediate][target]
                        if total_latency < min_latency:
                            min_latency = total_latency
                            corresponding_bandwidth = min(
                                source_to_node_bandwidth[s][intermediate],
                                bandwidth_matrix[intermediate][target]
                            )
                equivalent_latency[s][target] = min_latency
                equivalent_bandwidth[s][target] = corresponding_bandwidth

        return equivalent_latency, equivalent_bandwidth

    def calculate_task_trigger_frequency(self):
        """
        计算每个任务的触发频率，基于任务依赖关系（adjacency_matrix）和触发源的触发频率。
        """
        num_tasks = len(self.task_info["task_node_values"])
        num_sources = self.task_info["num_sources"]
        adjacency_matrix = self.task_info["adjacency_matrix"]
        sources_to_start_tasks = self.task_info["sources_to_start_tasks"]

        # 用于存储每个任务的触发频率
        task_trigger_frequency = [0] * num_tasks
        # Step 1: 根据 sources_to_start_tasks 矩阵计算初始任务的触发频率
        for source_id in range(num_sources):
            for task_id in range(num_tasks):
                if sources_to_start_tasks[task_id][source_id] > 0:  # 触发源对该任务有作用
                    trigger_info = self.resource_info["trigger_types"][source_id]
                    if trigger_info["type"] == "Normal":
                        frequency = 1 / max(trigger_info["mean"], 1e-6)
                    elif trigger_info["type"] == "Periodic":
                        frequency = 1 / max(trigger_info["T"], 1e-6)
                    elif trigger_info["type"] == "Poisson":
                        frequency = trigger_info["lambda"]
                    else:
                        raise ValueError(f"未知的触发类型: {trigger_info['type']}")

                    # 累加触发源对任务的触发频率（如果有多个触发源同时触发一个任务）
                    task_trigger_frequency[task_id] += frequency

        # Step 2: 进行拓扑排序，确保任务依赖关系有序
        from collections import deque

        # 计算每个任务的入度
        in_degree = [0] * num_tasks
        for i in range(num_tasks):
            for j in range(num_tasks):
                if adjacency_matrix[i][j] > 0:
                    in_degree[j] += 1

        # 将所有入度为 0 的任务加入队列
        queue = deque([i for i in range(num_tasks) if in_degree[i] == 0])

        # 使用队列计算触发频率
        while queue:
            current_task = queue.popleft()

            # 遍历当前任务的所有后续任务
            for next_task in range(num_tasks):
                if adjacency_matrix[current_task][next_task] > 0:  # 有依赖关系
                    # 累加当前任务对后续任务的触发频率贡献
                    task_trigger_frequency[next_task] += task_trigger_frequency[current_task]

                    # 更新后续任务的入度
                    in_degree[next_task] -= 1
                    if in_degree[next_task] == 0:
                        queue.append(next_task)

        # 检查是否存在环
        if any(in_degree):
            raise ValueError("任务依赖图中存在循环依赖，无法计算触发频率！")

        return task_trigger_frequency

    def calculate_sink_to_node_metrics(self, latency_matrix, sink_to_node_latency, bandwidth_matrix,
                                       sink_to_node_bandwidth):
        """计算sink到node的等效时延和带宽"""
        return self.calculate_source_to_node_metrics(
            latency_matrix, sink_to_node_latency, bandwidth_matrix, sink_to_node_bandwidth
        )

    def calculate_queueing_delay(self, task_id, node_id, task_priorities, allocation_matrix, execute_time,
                                 current_task_flow):
        """
        计算任务的排队时延
        Parameters:
        -----------
        task_id : int
            当前任务的编号
        node_id : int
            当前任务的分配节点
        task_priorities : list
            所有任务的优先级
        allocation_matrix : list[list[int]]
            任务分配矩阵
        execute_time : float
            当前任务的执行时间
        current_task_flow : dict
            当前任务流的信息，包含任务流中的任务编号等

        Returns:
        --------
        float
            任务的排队时延
        """
        # 当前任务的优先级
        current_priority = task_priorities[task_id]

        # 获取同节点上比当前任务优先级更高的任务
        higher_priority_tasks = [
            j for j in range(len(allocation_matrix[node_id]))
            if allocation_matrix[node_id][j] == 1 and task_priorities[j] > current_priority
        ]

        # 排除当前任务流内的任务
        same_flow_tasks = set(current_task_flow["task_in_path"])
        higher_priority_tasks = [j for j in higher_priority_tasks if j not in same_flow_tasks]

        # 计算排队时延
        total_occupancy = 0
        for task_j in higher_priority_tasks:
            execution_time_j = self.task_info["task_node_values"][task_j] / self.resource_info["compute_power"][node_id]
            total_occupancy += execution_time_j * self.task_trigger_frequency[task_j]

        idle_rate = 1 - total_occupancy
        if idle_rate <= 0:
            # raise ValueError(f"节点{node_id}被占满")
            print(f"节点{node_id}被占满!")
            return 0.3  # 显著大于一般时延
        queueing_delay = execute_time / idle_rate - execute_time
        return queueing_delay

    def simulate_key_task_flow_latency(self):
        """仿真所有关键任务流的总时延"""
        compute_power = self.resource_info["compute_power"]
        latency_matrix = self.resource_info["latency_matrix"]
        bandwidth_matrix = self.resource_info["bandwidth_matrix"]

        task_node_values = self.task_info["task_node_values"]
        adjacency_matrix = self.task_info["adjacency_matrix"]
        sources_to_start_tasks = self.task_info["sources_to_start_tasks"]
        end_tasks_to_sinks = self.task_info["end_tasks_to_sinks"]
        task_flows = self.task_info["key_task_flows"]

        allocation_matrix = self.task_allocation_info["allocation_matrix"]
        task_priorities = self.task_allocation_info["task_priorities"]

        task_flow_latencies = {}

        for flow_id, task_flow in enumerate(task_flows):
            task_path = task_flow["task_in_path"]
            source = task_flow["source"]
            sink = task_flow["sink"]

            total_latency = 0
            total_package_latency = 0
            total_transfer_latency = 0
            total_compute_latency = 0
            total_task_queueing_latency = 0

            # Step 1: 计算从 Source 到第一个任务节点的时延
            first_task = task_path[0]
            for node_id in range(len(allocation_matrix)):
                if allocation_matrix[node_id][first_task] == 1:
                    data_size = sources_to_start_tasks[first_task][source]
                    transfer_delay = data_size / self.resource_info['source_to_node_bandwidth'][source][node_id]
                    source_latency = self.resource_info["source_to_node_latency"][source][node_id]
                    total_latency += source_latency + transfer_delay
                    total_package_latency += source_latency
                    total_transfer_latency += transfer_delay
                    break

            # Step 2: 计算每个任务的计算时延和排队时延
            for i in range(len(task_path)):
                current_task = task_path[i]
                current_task_node = -1

                # 找到当前任务的分配节点
                for node_id in range(len(allocation_matrix)):
                    if allocation_matrix[node_id][current_task] == 1:
                        current_task_node = node_id
                        break
                if current_task_node != -1:
                    # 计算任务的执行时延
                    execute_time = task_node_values[current_task] / compute_power[current_task_node]
                    total_latency += execute_time
                    total_compute_latency += execute_time

                    # 计算任务的排队时延
                    queueing_delay = self.calculate_queueing_delay(
                        current_task, current_task_node, task_priorities, allocation_matrix, execute_time, task_flow
                    )
                    total_latency += queueing_delay
                    total_task_queueing_latency += queueing_delay
                else:
                    raise ValueError("存在未分配的节点")

                # Step 3: 计算任务之间的通信时延
                if i < len(task_path) - 1:
                    next_task = task_path[i + 1]
                    next_task_node = -1
                    for node_id in range(len(allocation_matrix)):
                        if allocation_matrix[node_id][next_task] == 1:
                            next_task_node = node_id
                            break
                    if current_task_node != -1 and next_task_node != -1:
                        transfer_data_size = adjacency_matrix[current_task][next_task]
                        transfer_delay = transfer_data_size / bandwidth_matrix[current_task_node][next_task_node]
                        package_delay = latency_matrix[current_task_node][next_task_node]
                        total_latency += package_delay + transfer_delay
                        total_package_latency += package_delay
                        total_transfer_latency += transfer_delay
                    else:
                        # 因为有前置条件 i < len(task_path) - 1，所以next_task_node不应该为-1
                        raise ValueError("存在未分配的节点")

            # Step 4: 计算从最后一个任务到 Sink 的时延
            last_task = task_path[-1]
            for node_id in range(len(allocation_matrix)):
                if allocation_matrix[node_id][last_task] == 1:
                    data_size = end_tasks_to_sinks[last_task][sink]
                    transfer_delay = data_size / self.resource_info['sink_to_node_bandwidth'][sink][node_id]
                    sink_latency = self.resource_info["sink_to_node_latency"][sink][node_id]
                    total_latency += sink_latency + transfer_delay
                    total_package_latency += sink_latency
                    total_transfer_latency += transfer_delay
                    break

            task_flow_latencies[flow_id] = {
                "total_latency": total_latency,
                "total_package_latency": total_package_latency,
                "total_transfer_latency": total_transfer_latency,
                "total_compute_latency": total_compute_latency,
                "total_task_queueing_latency": total_task_queueing_latency
            }

        return task_flow_latencies

    def run_simulation(self):
        """运行仿真并输出结果"""
        latencies = self.simulate_key_task_flow_latency()
        for flow_id, latency_info in latencies.items():
            print(f"flow {flow_id}'s total latency: {latency_info['total_latency']}")
            print(f"total_package_latency: {latency_info['total_package_latency']}")
            print(f"total_transfer_latency: {latency_info['total_transfer_latency']}")
            print(f"total_compute_latency: {latency_info['total_compute_latency']}")
            print(f"total_task_queueing_latency: {latency_info['total_task_queueing_latency']}")


if __name__ == "__main__":
    # 使用示例
    simulator = TaskFlowSimulator()
    simulator.load_resource_info('TTFM_data/computing_network_info.json')
    simulator.load_task_info('TTFM_data/task_info.json')
    simulator.load_allocation_info('TTFM_data/allocation_and_priority.json')

    simulator.run_simulation()
