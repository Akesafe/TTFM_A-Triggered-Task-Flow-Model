## Environment Setup

```bash
pip install -r requirements.txt
```

This project has been tested in the following environment:

*   **CPU:** Intel(R) Xeon(R) W-2125 CPU @ 4.00GHz
*   **RAM:** 32GB DDR4 2666MHz
*   **GPU:** NVIDIA GeForce RTX 3060
*   **OS:** Windows Server 2022 Standard
*   **Python Version:** 3.10.15

## Generating Task and Computing Network Models

Run `Experiment_Setup/Model_Generation/Convert_STG_to_TTFM_Task.py` to convert the STG model from `Experiment/STG_data/stg_model.stg` into a TTFM task model, which will be saved in `TTFM_data/task_info.json`.

Run `Experiment_Setup/Model_Generation/Generate_TTFM_Resource.py` to generate a TTFM computing network model conforming to a "Cloud-Edge-Device" architecture, which will be saved in `TTFM_data/computing_network_info.json`.

## Model Testing

Run `Experiment_test/random_task_schedule.py` to generate a random allocation and scheduling policy, which will be saved in `TTFM_data/allocation_and_priority.json`.

Then, run `TTFM_Simulation_main.py`. It will load `task_info.json` and `computing_network_info.json` from the `TTFM_data` directory, perform format checks, and then read `allocation_and_priority.json` to run the simulation and print the results.

The `TTFM_Simulation_main.py` file is the core file of the entire project and is called by various algorithms to perform simulations.

If you modify relevant parameters (e.g., changing the task scale or fine-tuning algorithm parameters) or regenerate intermediate files, the specific output of the program may change. However, this typically does not affect the overall trends or final conclusions. This also applies to the subsequent experiments.

### Exhaustive Search

Run `Experiment_test/search_test.py`. It will load the task and computing network models and perform an exhaustive search over all possible scheduling policies. **Note:** When the number of tasks or nodes is greater than 5, the program's runtime will become excessively long (over 10 hours) and it will exit automatically. It is recommended to use a small-scale task and computing node setup when running this test.

### Single Heuristic Algorithm Test

In the `Optimization_Methods` directory, you can run `xx_Algorithms.py` to execute a single round of a heuristic search algorithm.

## Experiment 1: Delay Composition Analysis
Run `Experiment/exp1_Delay_Composition_Analysis.py`.
It generates different scenarios (varying by the ratio of the number of computing tasks to computing nodes) and compares the differences in their delay composition, including computation delay, communication delay, and queuing delay.

## Experiment 2: Heuristic Algorithm Comparison

Run `Experiment/exp2_Alg_Comparison/Draw_all_global_fitness.py`.
When this script runs, it will first attempt to read a CSV file from the `result` directory and use it to generate plots.
If the file is not found, it will load the task and computing network models, call the four heuristic algorithms from `Optimization_Methods`, save the results as a CSV file in the `result` directory, and then generate the plots.

### Single Instance Run of TS Algorithm

Run `Discussion/Draw_single_instance_test.py` to visualize the fitness changes during the search process of a single Tabu Search instance.

## Experiment 3: Performance Analysis of Tabu Search Algorithm at Different Scales

Run `Experiment/exp3_Complexity_Analysis/Complexity_Analysis.py`.
When this script runs, it will first attempt to load models or results from files within the `scale_n` directory.
If the files are not found, it will create these directories, generate models for the corresponding scales, call the Tabu Search algorithm, and generate plots. This program typically has a long runtime. If it is interrupted, you can restart it, and it will resume from the breakpoint by reading from the `scale_n` directory.

## Discussion: Discrete Event Queuing Simulation Program
Run `Discussion/simulate_for_rho-t.py`.
It generates interference programs with different occupancy rates based on given parameters and calculates the queuing time of the target program under such interference. It repeats the process multiple times to draw scatter plots and compares them with theoretical time.

## Other Notes

You can modify various parameters or create new script files for other research and testing purposes based on this project. It is not recommended to change the names or locations of existing files, as this may break the calling relationships.


If you use this code in your research, please cite our paper:

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

If you have any questions or encounter any issues, you can contact us at this email address:

*akesafe@qq.com*
