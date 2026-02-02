import os
import sys
import json
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# --- Add project root to Python path ---
# This assumes the script is run from within the Optimization_Methods/Complexity_Analysis directory
# Or that the necessary modules are otherwise findable in the PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../Optimization_Methods', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Add project root ---

# --- Import necessary modules from the project ---

from Experiment_Setup.Model_Generation.Generate_TTFM_Task_RANDOM import TaskDAG, save_json_compact_lists
from Experiment_Setup.Model_Generation.Generate_TTFM_Resource import NetworkGenerator
from TTFM_Simulation_main import TaskFlowSimulator
from Optimization_Methods.TS_Algorithms import TabuSearch

# --- End Import ---

# 设置matplotlib参数，使图形更适合学术论文 (Copied from Draw_global_fitness.py)
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6) # Adjusted for potentially wider box plot
})


# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALES = [10, 30, 100, 300] # Problem scales (number of tasks)
NODE_TASK_RATIO = 0.5 # num_nodes = num_tasks * ratio
MASTER_SEED = 2025 # Master seed for reproducibility across scales if needed

# --- Parameters for Finding Best Known Solution ---
BEST_KNOWN_ITER_FACTOR = 30 # max_iterations = factor * num_tasks
BEST_KNOWN_SEED = 114515 # Fixed seed for finding best known solution

# --- Parameters for Search Runs ---
NUM_SEARCH_RUNS = 30 # Number of times to run TS for each scale to collect iteration data
SEARCH_MAX_ITER_FACTOR = 5 # Max iterations for search runs = factor * num_tasks (cap might be needed)
SEARCH_MAX_ITER_CAP = 50000 # Absolute cap on search iterations
TARGET_FITNESS_TOLERANCE = 1.1 # Target fitness = best_known * tolerance (10%)

# --- Consistent Tabu Search Parameters ---
# Using relatively small values for faster execution, adjust if needed
# Consider scaling these with problem size too, but keeping fixed for now
TS_PARAMS = {
    "tabu_tenure": 15,           # Maybe scale this? e.g., max(10, num_tasks // 10)
    "neighborhood_size": 15,     # Maybe scale this? e.g., max(15, num_tasks // 5)
    "diversification_threshold": 15  # Maybe scale this? e.g., max(20, num_tasks // 4)
}

# --- Helper Functions ---
def ensure_dir(directory):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def set_random_seeds(seed):
    """Sets random seeds for Python, NumPy."""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Set random seeds to: {seed}")

# --- Main Logic ---
all_results_data = [] # To store data for final CSV and plotting

start_time_total = time.time()

for scale in SCALES:
    print(f"\n--- Processing Scale: {scale} Tasks ---")
    scale_start_time = time.time()
    num_tasks = scale
    num_nodes = max(2, int(num_tasks * NODE_TASK_RATIO)) # Ensure at least 2 nodes
    scale_dir = os.path.join(BASE_DIR, f"scale_{scale}")
    ensure_dir(scale_dir)

    # Define file paths for this scale
    task_info_path = os.path.join(scale_dir, f"task_info_{scale}.json")
    resource_info_path = os.path.join(scale_dir, f"resource_{scale}.json")
    best_known_path = os.path.join(scale_dir, f"best_known_solution_{scale}.json")
    iteration_results_path = os.path.join(scale_dir, f"iteration_results_{scale}.csv")

    # --- Step 1: Generate Problem Data (if needed) ---
    scale_generation_seed = MASTER_SEED + scale # Unique seed per scale
    if not os.path.exists(task_info_path) or not os.path.exists(resource_info_path):
        print(f"Generating task and resource files for scale {scale}...")
        set_random_seeds(scale_generation_seed)
        try:
            # Generate Task Info
            task_dag = TaskDAG(num_of_task=num_tasks, random_seed=scale_generation_seed)
            task_dag.generate_dag()
            task_dag.save_output(task_info_path)
            print(f"Saved task info to: {task_info_path}")

            # Generate Resource Info
            network_generator = NetworkGenerator(num_of_compute_node=num_nodes, random_seed=scale_generation_seed)
            network_generator.generate_network(task_info_path=task_info_path, output_path=resource_info_path)
            print(f"Saved resource info to: {resource_info_path}")
            # Optional: Visualize generated network/DAG for debugging
            # task_dag.draw_dag()
            # network_generator.visualize_network()

        except Exception as e:
            print(f"Error generating data for scale {scale}: {e}")
            continue # Skip to next scale
    else:
        print(f"Task and resource files found for scale {scale}.")

    # --- Step 2: Find Best-Known Solution (if needed) ---
    best_known_fitness = None
    if not os.path.exists(best_known_path):
        print(f"Finding best-known solution for scale {scale}...")
        find_best_start_time = time.time()
        try:
            # Load simulator
            simulator = TaskFlowSimulator()
            simulator.load_resource_info(resource_info_path)
            simulator.load_task_info(task_info_path)

            # Configure and run TS with many iterations
            set_random_seeds(BEST_KNOWN_SEED) # Use fixed seed for this step
            max_iter_best_known = BEST_KNOWN_ITER_FACTOR * num_tasks
            print(f"Running TS with max_iterations = {max_iter_best_known} to find best-known solution...")

            # Adjust TS Params based on scale if desired (example below)
            # current_ts_params = TS_PARAMS.copy()
            # current_ts_params["tabu_tenure"] = max(10, num_tasks // 10)
            # current_ts_params["neighborhood_size"] = max(15, num_tasks // 5)
            # current_ts_params["diversification_threshold"] = max(20, num_tasks // 4)
            current_ts_params = TS_PARAMS.copy() # Using fixed params for now

            ts_best = TabuSearch(
                simulator=simulator,
                num_tasks=num_tasks,
                num_nodes=num_nodes,
                max_iterations=max_iter_best_known,
                **current_ts_params
            )
            ts_best.run()
            best_known_fitness = ts_best.best_fitness
            best_known_solution = ts_best.best_solution # Optional to save

            # Save the result
            result_data = {
                "best_known_fitness": best_known_fitness,
                "num_tasks": num_tasks,
                "num_nodes": num_nodes,
                "max_iterations_used": max_iter_best_known,
                # "best_solution": best_known_solution # Uncomment if needed
            }
            with open(best_known_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            print(f"Saved best-known fitness ({best_known_fitness:.6f}) to: {best_known_path}")
            find_best_duration = time.time() - find_best_start_time
            print(f"Time to find best-known solution: {find_best_duration:.2f} seconds")

        except Exception as e:
            print(f"Error finding best-known solution for scale {scale}: {e}")
            continue # Skip to next scale
    else:
        print(f"Loading best-known solution from: {best_known_path}")
        try:
            with open(best_known_path, 'r') as f:
                result_data = json.load(f)
            best_known_fitness = result_data["best_known_fitness"]
            print(f"Loaded best-known fitness: {best_known_fitness:.6f}")
        except Exception as e:
            print(f"Error loading best-known solution file {best_known_path}: {e}")
            continue

    if best_known_fitness is None:
        print(f"Could not determine best-known fitness for scale {scale}. Skipping search runs.")
        continue

    # --- Step 3: Run Multiple Searches for Iteration Count ---
    iterations_to_target = []
    run_data_exists = False
    max_iter_search = min(SEARCH_MAX_ITER_FACTOR * num_tasks, SEARCH_MAX_ITER_CAP)
    if os.path.exists(iteration_results_path):
        try:
            temp_df = pd.read_csv(iteration_results_path)
            if len(temp_df) >= NUM_SEARCH_RUNS:
                print(f"Sufficient iteration results found in: {iteration_results_path} ({len(temp_df)} runs). Loading...")
                iterations_to_target = temp_df['IterationsToTarget'].tolist()
                run_data_exists = True
            else:
                print(f"Found {len(temp_df)} runs in {iteration_results_path}, need {NUM_SEARCH_RUNS}. Re-running.")
        except Exception as e:
            print(f"Error reading {iteration_results_path}, will regenerate. Error: {e}")

    if not run_data_exists:
        print(f"Running {NUM_SEARCH_RUNS} searches for scale {scale} to find iterations needed...")
        search_runs_start_time = time.time()
        try:
            # Load simulator (might be needed again if not loaded above)
            simulator = TaskFlowSimulator()
            simulator.load_resource_info(resource_info_path)
            simulator.load_task_info(task_info_path)

            target_fitness = best_known_fitness * TARGET_FITNESS_TOLERANCE

            print(f"Target fitness (<= {TARGET_FITNESS_TOLERANCE*100:.1f}%): {target_fitness:.6f}")
            print(f"Max iterations per search run: {max_iter_search}")

            current_ts_params = TS_PARAMS.copy() # Use fixed params

            for run_num in range(NUM_SEARCH_RUNS):
                run_seed = MASTER_SEED + scale * 1000 + run_num # Unique seed per run
                set_random_seeds(run_seed)

                ts_run = TabuSearch(
                    simulator=simulator,
                    num_tasks=num_tasks,
                    num_nodes=num_nodes,
                    max_iterations=max_iter_search,
                    **current_ts_params
                )
                ts_run.run()

                # Analyze log data to find first iteration reaching target
                iter_found = -1 # Use -1 to indicate not found within max_iter_search
                for log_entry in ts_run.log_data:
                    # Skip non-numeric iterations or fitness values
                    if not isinstance(log_entry["Iteration"], int) or \
                       not isinstance(log_entry["Best_Fitness_Global"], (int, float)):
                        continue

                    current_iter = log_entry["Iteration"]
                    current_best_global = log_entry["Best_Fitness_Global"]

                    if current_best_global <= target_fitness:
                        iter_found = current_iter
                        break # Found the first time it met the target

                iterations_to_target.append(iter_found)
                if (run_num + 1) % 5 == 0 or run_num == NUM_SEARCH_RUNS - 1:
                     print(f"  Run {run_num + 1}/{NUM_SEARCH_RUNS} completed. Iterations to target: {iter_found}")


            # Save iteration results to CSV
            results_df = pd.DataFrame({'Run': range(1, NUM_SEARCH_RUNS + 1), 'IterationsToTarget': iterations_to_target})
            results_df.to_csv(iteration_results_path, index=False)
            print(f"Saved iteration results to: {iteration_results_path}")
            search_runs_duration = time.time() - search_runs_start_time
            print(f"Time for {NUM_SEARCH_RUNS} search runs: {search_runs_duration:.2f} seconds")

        except Exception as e:
            print(f"Error during search runs for scale {scale}: {e}")
            continue # Skip scale

    # --- Store results for final aggregation ---
    if iterations_to_target:
        # Filter out runs that didn't reach the target (-1) before adding
        valid_iterations = [it for it in iterations_to_target if it != -1]
        num_failed = len(iterations_to_target) - len(valid_iterations)
        if num_failed > 0:
            print(f"Scale {scale}: {num_failed}/{NUM_SEARCH_RUNS} runs did not reach the target within {max_iter_search} iterations.")

        if valid_iterations: # Only add if at least one run succeeded
             for it in valid_iterations:
                 all_results_data.append({'Scale': scale, 'Iterations': it})
        else:
             print(f"Scale {scale}: No runs reached the target fitness.")
    else:
         print(f"Scale {scale}: No iteration data collected.")


    scale_duration = time.time() - scale_start_time
    print(f"--- Scale {scale} processing time: {scale_duration:.2f} seconds ---")


# --- Step 4: Aggregate and Plot ---
print("\n--- Aggregating Results and Plotting ---")

if not all_results_data:
    print("No results data collected. Exiting.")
    sys.exit(1)

final_df = pd.DataFrame(all_results_data)

# Save aggregated data to CSV
final_csv_path = os.path.join(BASE_DIR, "complexity_analysis_iterations.csv")
final_df.to_csv(final_csv_path, index=False)
print(f"Saved aggregated data for plotting to: {final_csv_path}")

# Create the box plot
plt.figure(dpi=300) # Use settings from Draw_global_fitness.py
# Prepare data for boxplot: list of arrays/lists, one for each scale
plot_data = [final_df[final_df['Scale'] == s]['Iterations'].values for s in SCALES if s in final_df['Scale'].unique()]
scale_labels = [str(s) for s in SCALES if s in final_df['Scale'].unique()] # Labels for x-axis

if not plot_data:
     print("No valid data to plot.")
     sys.exit(1)

plt.boxplot(plot_data, labels=scale_labels, showfliers=False, showmeans=True) # showfliers=False to hide outliers
plt.yscale('log')
plt.xlabel('Problem Scale (Number of Tasks)')
plt.ylabel(f'Iterations to Reach {TARGET_FITNESS_TOLERANCE*100:.0f}% of Best Known')
plt.title('TS Algorithm Complexity: Iterations vs Problem Scale')
plt.grid(True, linestyle='--', alpha=0.7, which='both')
plt.tight_layout()

# Save the plot
plot_path = os.path.join(BASE_DIR, "complexity_analysis_boxplot.png")
plt.savefig(plot_path, bbox_inches='tight')
print(f"Saved box plot to: {plot_path}")

# Show the plot
plt.show()

# =========================================================================
# --- New Section: Measure Single Evaluation Time ---
# =========================================================================
print("\n--- Measuring Single Evaluation Time for TTFM ---")
eval_time_results = []
TIMING_ITERATIONS = 100
TIMING_NEIGHBORHOOD_SIZE = 10
TOTAL_EVALS_FOR_TIMING = TIMING_ITERATIONS * TIMING_NEIGHBORHOOD_SIZE  # 100 * 10 = 1000

for scale in SCALES:
    scale_dir = os.path.join(BASE_DIR, f"scale_{scale}")
    task_info_path = os.path.join(scale_dir, f"task_info_{scale}.json")
    resource_info_path = os.path.join(scale_dir, f"resource_{scale}.json")

    # Double check file existence
    if not os.path.exists(task_info_path) or not os.path.exists(resource_info_path):
        print(f"Skipping timing for scale {scale} (files not found)")
        continue

    # 1. Load Simulator for this scale
    simulator = TaskFlowSimulator()
    simulator.load_resource_info(resource_info_path)
    simulator.load_task_info(task_info_path)

    # Recalculate node count for TS init
    num_tasks_current = len(simulator.task_info["task_node_values"])
    num_nodes_current = len(simulator.resource_info["compute_power"])

    # 2. Configure TS for timing run
    # Iterations=100, Neighborhood=10 -> Approx 1000 Fitness Evaluations
    ts_timing = TabuSearch(
        simulator=simulator,
        num_tasks=num_tasks_current,
        num_nodes=num_nodes_current,
        max_iterations=TIMING_ITERATIONS,
        tabu_tenure=10,  # Keep small for consistency
        neighborhood_size=TIMING_NEIGHBORHOOD_SIZE,
        diversification_threshold=TIMING_ITERATIONS + 5  # Disable diversification
    )

    # 3. Measure Execution Time
    t_start = time.time()
    ts_timing.run()
    t_end = time.time()

    duration = t_end - t_start

    # 4. Calculate Single Evaluation Time (Duration / 1000)
    avg_eval_time = duration / TOTAL_EVALS_FOR_TIMING

    print(f"Scale {scale}: Total Time (1000 evals) = {duration:.4f}s, Single Eval = {avg_eval_time:.8f}s")

    eval_time_results.append({
        "Scale": scale,
        "Total_Duration_s": duration,
        "Single_Evaluation_Time_s": avg_eval_time
    })

# 5. Save to CSV
eval_csv_path = os.path.join(BASE_DIR, "ttfm_single_evaluation_time.csv")
pd.DataFrame(eval_time_results).to_csv(eval_csv_path, index=False)
print(f"Saved evaluation timing results to: {eval_csv_path}")
# =========================================================================


total_duration = time.time() - start_time_total
print(f"\nTotal analysis runtime: {total_duration:.2f} seconds")
print("Complexity Analysis Script Finished.")