import numpy as np
import matplotlib.pyplot as plt
import random
import math

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==========================================
# 1. 仿真参数配置
# ==========================================
PROCESSOR_R = 100.0  # 处理器算力 R (MIPS)
T0_LOAD = 1  # 测试程序 t0 的负载 (MI)
C0 = T0_LOAD / PROCESSOR_R  # t0 单独运行需要的纯执行时间 (秒)

NUM_INTERFERENCE_TASKS = 100  # 每次生成由多个不同的干扰任务组成的集合
MIN_FREQ = 20  # 干扰任务最小频率
MAX_FREQ = 200  # 干扰任务最大频率

# 绘图配置
RHOS_TO_TEST = np.arange(0, 0.96, 0.02)
REPEAT_TIMES = 10


# ==========================================
# 2. 核心计算函数
# ==========================================
class Task:
    def __init__(self, load, freq, offset=0.0):
        self.load = load
        self.freq = freq
        self.period = 1.0 / freq
        self.cost = load / PROCESSOR_R
        self.offset = offset


def calculate_response_time(t0_cost, interference_tasks):
    t = t0_cost
    while True:
        interference_work = 0.0
        for task in interference_tasks:
            if t > task.offset:
                num_jobs = math.ceil((t - task.offset) / task.period)
                interference_work += num_jobs * task.cost
        new_t = t0_cost + interference_work
        if abs(new_t - t) < 1e-9:
            return new_t
        t = new_t
        if t > 1000 * t0_cost: return t


# ==========================================
# 3. 主仿真循环
# ==========================================
sim_rho_x = []
sim_time_y = []
avg_rho_x = []
avg_time_y = []

print(f"开始仿真 (High Freq Mode: {MIN_FREQ}-{MAX_FREQ}Hz)...")

for target_rho in RHOS_TO_TEST:
    if target_rho >= 1.0: continue

    current_batch_times = []

    for _ in range(REPEAT_TIMES):
        current_tasks = []
        initial_blocking = 0.0

        if target_rho > 0:
            # 1. 随机生成任务组合 (Task Set Generation)
            proportions = np.random.dirichlet(np.ones(NUM_INTERFERENCE_TASKS))
            utilizations = proportions * target_rho

            for u in utilizations:
                f = random.uniform(MIN_FREQ, MAX_FREQ)
                period = 1.0 / f
                # Cost = Utilization * Capacity / Frequency
                # 注意：Task类接收的是 Load (MI)，不是 Cost (s)
                # Cost (s) = u * R / f / R = u / f
                # Load (MI) = Cost * R = (u / f) * R
                task_load = (u * PROCESSOR_R) / f
                cost = u / f

                # 2. 随机生成相位 (Phase Generation)
                offset = random.uniform(0, period)

                current_tasks.append(Task(task_load, f, offset))

                # 3. 计算 Carry-in
                prev_finish_time = (offset - period) + cost
                if prev_finish_time > 0:
                    initial_blocking += prev_finish_time

        # 计算
        actual_time = calculate_response_time(C0 + initial_blocking, current_tasks)
        queue_delay = actual_time - C0

        sim_rho_x.append(target_rho)
        sim_time_y.append(queue_delay)
        current_batch_times.append(queue_delay)

    avg_rho_x.append(target_rho)
    avg_time_y.append(np.mean(current_batch_times))

print("仿真结束，正在绘图...")

# avg_time_y 包含了每个 rho 对应的原始平均响应时间
# 我们现在对其进行平滑处理，生成一个新的列表 moving_avg_y
moving_avg_y = []
# 定义窗口半径：当前点的前后各取2个点
window_radius = 1
for i in range(len(avg_time_y)):
    if i < window_radius or i >= len(avg_time_y) - window_radius:
        moving_avg_y.append(avg_time_y[i])
    else:
        # 只有在数据充足的中间段，才进行滑动平均
        start_index = i - window_radius
        end_index = i + window_radius + 1
        window_data = avg_time_y[start_index:end_index]
        moving_avg_y.append(np.mean(window_data))

# ==========================================
# 4. 结果绘图
# ==========================================
plt.figure(figsize=(6, 4))
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # 让数学公式符号也接近 Times 风格

theory_rho = np.linspace(0, 0.96, 200)
theory_queue_delay = C0 * ((1 / (1 - theory_rho)) - 1)

plt.plot(theory_rho, theory_queue_delay , 'r-', linewidth=2, label='Theoretical Prediction', zorder=10)

plt.scatter(sim_rho_x, sim_time_y, c='blue', alpha=0.3, s=8, label='Simulated Samples', zorder=1)
plt.plot(avg_rho_x, moving_avg_y, 'g--', linewidth=2, marker='.', markersize=4, label='Simulation Mean', zorder=15)

plt.title(
    f'Task Queuing Delay($\\tau_{{\mathrm{{queue}}}}$) vs. Processor Utilization ($\\rho$)\n'
    f'{NUM_INTERFERENCE_TASKS} interference tasks, Trigger Freq. $F_{{\mathrm{{Tri}}}} \\in [{MIN_FREQ}, {MAX_FREQ}]$ Hz',
    fontsize=12)
plt.xlabel('Processor Utilization ($\\rho$)', fontsize=12)
plt.ylabel('Queuing Delay ($\\tau_{\mathrm{{queue}}}$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=11)
plt.xlim(-0.02, 0.96)
# 限制一下Y轴，避免极端点把图拉得太长看不清趋势
plt.ylim(-max(theory_queue_delay)*0.05, max(theory_queue_delay))

plt.tight_layout()
plt.savefig(f"output/r={PROCESSOR_R}_t0={T0_LOAD}_under_{NUM_INTERFERENCE_TASKS}_{MIN_FREQ}-{MAX_FREQ}_Hz_interference.png")
plt.show()