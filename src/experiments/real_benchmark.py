import os
import pickle
from datetime import datetime
import pandas as pd

# Assume these are correctly imported from your project structure
from src.mcf_utils.single_results_utils import create_excel_results, create_kpi_comparison
from src.mcf_utils.loading_flow_utils import prepare_loader_flow_data, solve_loader_flow, solve_time_agg_loader_flow

print("Start - Single Instance Time Aggregation Experiment")

# --- 1. Configuration Setup ---

# Define the base file path for saving data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_file_path = os.path.join(BASE_DIR, "../../data/")

# Experiment names
experiments = ["benchmark_gurobi_TimeAgg", "benchmark_actual"]

# Instance size scenario definition
instance_size_scenario = {"nr_fields": 4133, "nr_h": 11, "nr_l": 11}
scenario_id = f"{instance_size_scenario['nr_fields']}_{instance_size_scenario['nr_h']}_{instance_size_scenario['nr_l']}"


# Gurobi Sensitivity Scenarios
sensitivity_scenarios = {"cost_multi": 1, "productivity_multi": 1, "working_hours_multi": 1}

# Gurobi Model Versions
gurobi_model_versions_2h = {"idle_time": True, "restricted": True, "access": False,
                            "working_hours": False, "travel_time": True, "t_p": 2}

gurobi_model_versions_1h = {"idle_time": True, "restricted": True, "access": False,
                            "working_hours": False, "travel_time": True, "t_p": 1}

# Time Aggregation Warm-start Configuration
time_agg_warmstart_params = {
    'type': 'selective_depot_related',
    'set_initial_depot_stays': True,
    'set_initial_depot_egress': True,
    'set_terminal_depot_stays': True,
}

# Solver and Model Parameters
solver_params = {
    "TimeLimit": 3600, "MIPGap": 0.001, "ScaleFlag": 2,
    "Seed": 42, "DualReductions": 0, "Threads": 0
}
vehicle_capacity_flag = True
restricted_flow_flag1h = False
restricted_flow_flag2h = False
last_period_restricted_flag1h = True
last_period_restricted_flag2h = True
add_min_beet_restriction_flag1h = False
add_min_beet_restriction_flag2h = False
v_type = 'binary'
verbose = False

# --- 2. Data Loading and Preparation ---

print(f"Loading instance data for scenario: {scenario_id}")
print("From: ../../data/results/instances/")

bench_file_1h_path = os.path.join(BASE_DIR, "../../data/results/instances/instance_data_benchmark_1h")
bench_instance_data_1h_full = pd.read_pickle(bench_file_1h_path)
instance_data_1h = bench_instance_data_1h_full[scenario_id]

bench_file_2h_path = os.path.join(BASE_DIR, "../../data/results/instances/instance_data_benchmark_2h")
bench_instance_data_2h_full = pd.read_pickle(bench_file_2h_path)
instance_data_2h = bench_instance_data_2h_full[scenario_id]

print("\nApplying custom data manipulations...")
# Custom data manipulations
instance_data_1h["loader_data"]["MeanProductivityPerHour"] = 200
instance_data_2h["loader_data"]["MeanProductivityPerHour"] = 200  # Apply to both

instance_data_1h["inventory_goal"] = instance_data_1h["production_volume_per_day"] * 2
instance_data_2h["inventory_goal"] = instance_data_2h["production_volume_per_day"] * 2

instance_data_1h["L"] = {72, 73, 74}
instance_data_2h["L"] = {72, 73, 74}
print("Custom manipulations applied successfully.")


# Extract heuristic inputs for both time resolutions
T_heuristic_1h = instance_data_1h["T_horizon_default"]
T_heuristic_2h = instance_data_2h["T_horizon_default"]
# Assuming routes are independent of time resolution, using 2h as primary
loader_routes_heuristic = instance_data_2h["loader_routes"]

# --- 3. Execute Benchmark Run (Enforced Schedule) ---

print("\n--- Running Benchmark (Enforced Actual Schedule) ---")

# Prepare data for the 1h benchmark model
instance_raw_1h_bench, derived_1h_bench = prepare_loader_flow_data(
    instance_data_1h,
    sens=sensitivity_scenarios,
    model_ver=gurobi_model_versions_1h,
    T_heuristic=T_heuristic_1h,
    loader_routes_heuristic=loader_routes_heuristic
)

instance_raw_2h_bench, derived_2h_bench = prepare_loader_flow_data(
    instance_data_2h,
    sens=sensitivity_scenarios,
    model_ver=gurobi_model_versions_2h,
    T_heuristic=T_heuristic_2h,
    loader_routes_heuristic=loader_routes_heuristic
)

combined_key = "real_benchmarking"

bench_results = {}

bench_results[combined_key] = solve_loader_flow(
    instance=instance_raw_1h_bench,
    derived=derived_1h_bench,
    name=experiments[1],
    hotstart_solution=None,
    enforce_solution=instance_data_1h,  # Enforce the schedule from the loaded data
    vehicle_capacity_flag=vehicle_capacity_flag,
    restricted_flow_flag=restricted_flow_flag1h,
    last_period_restricted_flag=last_period_restricted_flag1h,
    add_min_beet_restriction_flag=add_min_beet_restriction_flag1h,
    FIXED_SOLVER_PARAMS=solver_params,
    verbose=verbose,
    base_file_path=base_file_path,
    inventory_flag=True,
    inventory_cap_flag=False,
    v_type=v_type
)
print("--- Benchmark Run Finished 1h ---")

bench_results["2h"] = solve_loader_flow(
    instance=instance_raw_2h_bench,
    derived=derived_2h_bench,
    name="2h_test",
    hotstart_solution=None,
    enforce_solution=instance_data_2h,  # Enforce the schedule from the loaded data
    vehicle_capacity_flag=vehicle_capacity_flag,
    restricted_flow_flag=restricted_flow_flag2h,
    last_period_restricted_flag=last_period_restricted_flag2h,
    add_min_beet_restriction_flag=add_min_beet_restriction_flag2h,
    FIXED_SOLVER_PARAMS=solver_params,
    verbose=verbose,
    base_file_path=base_file_path,
    inventory_flag=True,
    inventory_cap_flag=False,
    v_type=v_type
)
print("--- Benchmark Run Finished 2h ---")


# --- 4. Execute Gurobi Time Aggregation Run ---

print("\n--- Running Gurobi Optimization with Time Aggregation ---")

gurobi_results_TimeAgg_1h = {}
# **This is the new call to the time aggregation function**
gurobi_results_TimeAgg_1h[combined_key] = solve_time_agg_loader_flow(
    # Data & Config
    instance_raw_coarse=instance_data_2h,
    instance_raw_fine=instance_data_1h,
    model_ver_coarse=gurobi_model_versions_2h,
    model_ver_fine=gurobi_model_versions_1h,
    sens_scn=sensitivity_scenarios,
    solver_params=solver_params,
    # Heuristic Inputs
    T_heuristic_coarse=T_heuristic_2h,
    T_heuristic_fine=T_heuristic_1h,
    loader_routes_heuristic=loader_routes_heuristic,
    # Time Aggregation Specific Params
    time_agg_warmstart_params=None,
    # General Model & Output Params
    base_file_path=base_file_path,
    combined_key=combined_key,
    name=experiments[0],
    # Hotstart 2h
    hotstart_solution2h=instance_data_2h,
    # Coarse Model Flags
    vehicle_capacity_flag_coarse=vehicle_capacity_flag,
    restricted_flow_flag_coarse=restricted_flow_flag2h,
    last_period_restricted_flag_coarse=last_period_restricted_flag2h,
    add_min_beet_restriction_flag_coarse=add_min_beet_restriction_flag2h,
    # Fine Model Flags
    vehicle_capacity_flag_fine=vehicle_capacity_flag,
    restricted_flow_flag_fine=restricted_flow_flag1h,
    last_period_restricted_flag_fine=last_period_restricted_flag1h,
    add_min_beet_restriction_flag_fine=add_min_beet_restriction_flag1h,
    # Common Flags
    v_type=v_type,
    verbose=verbose
)
print("--- Gurobi Time Aggregation Run Finished ---")


# --- 5. Save and Analyze Results ---

print("\nSaving results to pickle files...")
# Actual benchmarking results
bench_results_filepath = os.path.join(base_file_path, f"results/reporting/results_{experiments[1]}.pkl")
with open(bench_results_filepath, 'wb') as results_file:
    pickle.dump(bench_results, results_file)

# Gurobi Time Aggregation results
gurobi_results_filepath = os.path.join(base_file_path, f"results/reporting/results_{experiments[0]}.pkl")
with open(gurobi_results_filepath, 'wb') as results_file:
    pickle.dump(gurobi_results_TimeAgg_1h, results_file)
print("Results saved.")

print("\nCreating Excel summaries and KPI comparison...")
# Quick comparison Excel files
create_excel_results([combined_key], experiments, base_file_path, flow_type="load_flow")
print("Individual Excel summaries created.")

# KPI comparison
kpi_comparison = create_kpi_comparison(
    gurobi_results_TimeAgg_1h,
    bench_results,
    [combined_key],
    flow_type="load_flow"
)

# Save KPI comparison to Excel
today_date = datetime.now().strftime('%Y%m%d')
kpi_excel_file_path = os.path.join(
    BASE_DIR,
    f'../../data/results/excel/kpi_comparison_benchmarking_{today_date}.xlsx'
)
os.makedirs(os.path.dirname(kpi_excel_file_path), exist_ok=True)
kpi_comparison.to_excel(kpi_excel_file_path)
print(f"KPI comparison saved to: {kpi_excel_file_path}")

# Display KPI results in console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
print("\n--- KPI Comparison Results ---")

print(kpi_comparison.iloc[:, 1:])
