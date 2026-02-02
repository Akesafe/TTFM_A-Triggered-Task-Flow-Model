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
*(Paper link and citation format will be provided upon formal acceptance of the article.)*

If you have any questions or encounter any issues, you can contact us at this email address:
*(Contact information will be provided upon formal acceptance of the article.)*

Author's statement: This code must be made public after the corresponding paper is published and accompanied by the paper information. You are not allowed to disclose the preview version you have seen in any form. The original author reserves the right to pursue legal responsibility for the disclosure of the preview version.