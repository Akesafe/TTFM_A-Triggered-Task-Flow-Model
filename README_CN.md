## 配置环境

```bash
pip install -r requirements.txt
```
该项目已经在以下环境中进行了测试：

* **CPU**：Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz

* **RAM**：32GB DDR4 2666MHz

* **GPU**：NVIDIA GeForce RTX 3060

* **操作系统**：Windows Server 2022 Standard

* **Python版本**：3.10.15


## 生成任务和算网模型
运行`Experiment_Setup/Model_Generation/Convert_STG_to_TTFM_Task.py`，即可将`Experiment/STG_data/stg_model.stg`中的STG模型转化成TTFM任务模型，并保存在`TTFM_data/task_info.json`中；

运行`Experiment_Setup/Model_Generation/Generate_TTFM_Resource.py`，即可生成一个符合`云边端”架构的TTFM算网模型，并保存在`TTFM_data/computing_network_info.json`中。

## 模型测试
运行`Experiment_test/random_task_schedule.py`，将得到一个随机生成的分配调度策略，保存在`TTFM_data/allocation_and_priority.json`中。

然后运行`TTFM_Simulation_main.py`，它将载入`TTFM_data`中的`task_info.json`和`computing_network_info.json`，执行格式检查，然后读取`allocation_and_priority.json`运行仿真并打印结果。

`TTFM_Simulation_main.py`文件同时也是整个项目和核心文件，会被各种算法调用以用于执行仿真。

如果您改动了相关的参数（例如改变了任务规模/微调了某些算法参数等）或重新生成了某些中间文件，可能会导致程序的输出的具体结果发生变化，但这通常不影响整体趋势，也不影响最终结论。这一点对后面的实验同样适用。
### 遍历搜索
运行`Experiment_test/search_test.py`，它将载入任务模型和算网模型并对所有可能的调度策略执行遍历（注意：当任务数/节点数大于5时，程序运行时间将过长（大于10小时）因此会直接退出），运行此测试时，建议调整到一个较小的任务和计算节点规模。

### 单项启发式算法测试
在`Optimization_Methods`中可以运行`xx_Algorithms.py`运行一轮启发式搜索算法。

## 实验一：时延构成分析
运行`Experiment/exp1_Delay_Composition_Analysis.py`;
它会生成不同的场景（区别在于计算任务：计算节点的数量比例），并对比他们的时延构成差异（包括计算时延、通信时延、排队时延）。

## 实验二：启发式算法对比
运行`Experiment/exp2_Alg_Comparison/Draw_all_global_fitness.py`；
这个程序在运行时，会优先尝试读取`result`目录中的csv文件，然后用该文件生成绘图结果。
如果找不到该文件，则会尝试载入任务模型和算网模型，并调用`Optimization_Methods`中的四种启发式算法和，将结果保存为csv文件存放在`result`目录下并生成绘图结果。

### TS算法的单实例运行过程
运行`Discussion/Draw_single_instance_test.py`可以看到单个TS算法搜索实例的搜索过程中适应度的变化情况

## 实验三：禁忌搜索算法在不同规模下的性能测试
运行`Experiment/exp3_Complexity_Analysis/Complexity_Analysis.py`；
这个程序运行时，会优先尝试读取scale_n中的文件来载入模型，或者直接得到运行结果。
如果找不到文件，则会尝试生成这些目录，并创建对应规模的模型，调用禁忌搜索算法并生成绘图结果。这个程序的运行时间通常较长，如果运行到一半中断了，重新运行时仍然会读取scale_n目录，这样能够从断点处开始继续运行。

## 讨论：离散事件排队仿真程序
运行`Discussion/simulate_for_rho-t.py`；
它会根据给定的参数生成不同占用率的干扰程序，并计算目标程序在这种干扰下的排队时间，重复多次画出散点图，并于理论时间对比。

## 其他说明
基于该项目，您可以修改各种参数或新建程序文件来进行其他科研、测试。不建议更改现有程序文件的名称、位置，否则可能会破坏其调用关系。
如果您在研究中使用了该代码，请必须引用论文：

[*TTFM: A triggered task flow model for latency-critical task scheduling in distributed industrial systems*](https://doi.org/10.1016/j.cie.2026.111947)

```
@article{XIANG2026111947,
title = {TTFM: A triggered task flow model for latency-critical task scheduling in distributed industrial systems},
journal = {Computers \& Industrial Engineering},
volume = {215},
pages = {111947},
year = {2026},
issn = {0360-8352},
doi = {https://doi.org/10.1016/j.cie.2026.111947},
url = {https://www.sciencedirect.com/science/article/pii/S0360835226001488},
author = {Chengfeng Xiang and Na Chen and Lei Sun and Jianquan Wang and Ronghui Zhang and Zhangchao Ma},
keywords = {Task allocation, Latency optimization, Task scheduling, Distributed industrial system, Task flow model, IEC 61499},
abstract = {Optimizing task allocation and scheduling to minimize latency becomes increasingly critical for production efficiency and stability, as Industry 4.0 drives industrial systems to be more distributed and event-driven. However, conventional operations research and scheduling models often fail to capture the complex, event-driven characteristics and critical end-to-end process flows inherent to these decentralized automation systems. To address this gap, we propose the Triggered Task Flow Model (TTFM), a novel computational framework inspired by the IEC 61499 standard. TTFM provides a more realistic representation by explicitly modeling trigger sources, sinks, critical end-to-end task flows, and their quantifiable costs. To ground the model in practice, we establish a methodology for representing industrial applications and standard benchmarks within the TTFM framework. Analysis of latency composition verifies the model’s validity in capturing shifting physical bottlenecks. We then formulate the task allocation and scheduling optimization problem to minimize the latency of critical flows, which directly impacts production stability and responsiveness. A comparative evaluation of four heuristic algorithms identifies Tabu Search (TS) as providing a superior balance of solution quality and convergence speed for this NP-hard problem. Scalability analysis confirms the viability of TS for practical industrial scales. Our work provides industrial engineers and operations managers with a high-fidelity task model and its application methodology, validating its use for designing and managing next-generation, high-performance distributed industrial systems.}
}
```

如果有任何疑问或遇到任何问题，您可以联系此邮箱：
akesafe@qq.com
