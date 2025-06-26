"""
Create scripts where we set up the experiments and run three versions of 25-3 cases at -30% 0 + 30%
e.g. 140, 200, 260 productivity.

Variables
- Production Volume
- Productivity
    - Same as making fields bigger or smaller!
- Costs

Compare:
GAP Values, Complexity, Cost savings
"""

import os
from datetime import datetime
from src.data_prep.data_utils import raw_data_creation

from src.data_prep.data_utils import filter_instance_data

from src.mcf_utils.loading_flow_utils import run_pl_heuristic_experiments

from src.mcf_utils.loading_flow_utils import run_loader_flow_experiments
from src.mcf_utils.loading_flow_utils import run_loader_flow_TimeAgg

from src.mcf_utils.single_results_utils import *

print("Start - Robustness Test")

# Define the base file path for saving data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_file_path = os.path.join(BASE_DIR, "../../data/")

# Experiment names
experiments = ["robustness_g", "robustness_h"]

# Raw Data Generation
field_scenarios = [

    {"scenario": "300S", "n_fields": 300, "max_radius": 100}
]

# Instance Sizes scenario Definition
instance_size_scenarios = [{"nr_fields": 4133, "nr_h": 11, "nr_l": 11}]

filtered_size_scenarios = [
    {"nr_fields": 25, "nr_h": 2, "nr_l": 2}
]

# Gurobi Sensitivity Scenarios
sensitivity_scenarios = [

    # Costs
    {"cost_multi": 1, "productivity_multi": 1, "working_hours_multi": 1},
    {"cost_multi": 0.5, "productivity_multi": 1, "working_hours_multi": 1},
    {"cost_multi": 1.5, "productivity_multi": 1, "working_hours_multi": 1},

    # Productivity
    {"cost_multi": 1, "productivity_multi": 0.5, "working_hours_multi": 1},
    {"cost_multi": 1, "productivity_multi": 1.5, "working_hours_multi": 1},
]
# Gurobi Model Versions
gurobi_model_versions_2h = [

    {"idle_time": True, "restricted": True, "access": False,
     "working_hours": False, "travel_time": True, "t_p": 2},

]

gurobi_model_versions_1h = [

    {"idle_time": True, "restricted": True, "access": False,
     "working_hours": False, "travel_time": True, "t_p": 1},

]

FIXED_MODEL_CFG = {"v_type": "binary"}

solver_params = {
    "TimeLimit": 3600,
    "MIPGap": 0.001,
    "ScaleFlag": 2,
    "Seed": 42,
    "DualReductions": 0,
    "Threads": 0
}

# Define the new params dictionary
time_agg_params = {
    'vehicle_capacity_flag': False,
    'restricted_flow_flag1h': False,
    'restricted_flow_flag2h': False,
    'last_period_restricted_flag1h': True,     # Needs to be false
    'last_period_restricted_flag2h': True,      # We use more efficient restriction in coarse version
    'add_min_beet_restriction_flag1h': False,    # We like to use beet goals in fine version
    'add_min_beet_restriction_flag2h': False,   # Needs to be false
    'time_restriction': False,
    'verbose': False,
    'v_type': "binary"
}

print("Instance Data from: ../../data/results/instances/")

data_file_2h = os.path.join(BASE_DIR, '../../data/results/instances/simulated_instance_data_2h.pkl')
instance_data_2h = pd.read_pickle(data_file_2h)

data_file_1h = os.path.join(BASE_DIR, '../../data/results/instances/simulated_instance_data_1h.pkl')
instance_data_1h = pd.read_pickle(data_file_1h)

# Define the loader_ids and harvester_ids you want to filter by
loader_ids = [45, 73]
harvester_ids = [1, 3]
vehicle_capacity_flag = False

nr_fields = list(range(15, 36, 5))

print("Start Filter Data")

# Filter the instance_data's
filtered_instance_data_2h = filter_instance_data(
    instance_data=instance_data_2h,
    loader_ids=loader_ids,
    harvester_ids=harvester_ids,
    nr_fields_list=nr_fields,
    exclude_holidays=True,
    reschedule=True
)

# Filter the instance_data
filtered_instance_data_1h = filter_instance_data(
    instance_data=instance_data_1h,
    loader_ids=loader_ids,
    harvester_ids=harvester_ids,
    nr_fields_list=nr_fields,
    exclude_holidays=True,
    reschedule=True
)

# Run Heuristics 2h
pl_heuristic_results_2h = run_pl_heuristic_experiments(
    filtered_size_scenarios, filtered_instance_data_2h,
    sensitivity_scenarios, base_file_path,
    model_versions=gurobi_model_versions_2h,
    vehicle_capacity_flag=vehicle_capacity_flag,
    time_restriction=False,
    verbose=True,
    usage=True)

# Run Heuristic 1h
pl_heuristic_results_1h = run_pl_heuristic_experiments(
    filtered_size_scenarios, filtered_instance_data_1h,
    sensitivity_scenarios, base_file_path,
    model_versions=gurobi_model_versions_1h,
    vehicle_capacity_flag=vehicle_capacity_flag,
    time_restriction=False,
    verbose=True,
    usage=True)

# Get Benchmark 1h
heuristic_results_1h = run_loader_flow_experiments(
    filtered_size_scenarios, filtered_instance_data_1h,
    sensitivity_scenarios, gurobi_model_versions_1h,
    solver_params, base_file_path, experiments[1],
    hotstart_solution=None,
    enforce_solution=pl_heuristic_results_1h,
    vehicle_capacity_flag=False,
    restricted_flow_flag=False,
    last_period_restricted_flag=True,
    time_restriction=False,
    verbose=False,
    v_type=FIXED_MODEL_CFG["v_type"])

print("\n START TIME_AGG!! \n")

gurobi_results_TimeAgg_1h = run_loader_flow_TimeAgg(
    filtered_size_scenarios,
    filtered_instance_data_2h,
    filtered_instance_data_1h,
    sensitivity_scenarios,
    gurobi_model_versions_2h,
    gurobi_model_versions_1h,
    solver_params,
    base_file_path,
    experiments[0],
    hotstart_solution=pl_heuristic_results_2h,
    enforce_solution=None,
    params=time_agg_params
)

# Get today's date in yyyymmdd format
today_date = datetime.now().strftime('%Y%m%d')

# Quick comparison
combined_key_list = create_combined_key_list(filtered_size_scenarios, gurobi_model_versions_1h, sensitivity_scenarios)
create_excel_results(combined_key_list, experiments, base_file_path, flow_type="load_flow")
print("Excel Created")

kpi_comparison = create_kpi_comparison(gurobi_results_TimeAgg_1h, heuristic_results_1h,
                                       combined_key_list, flow_type="load_flow")

# Get Path
kpi_excel_file_path = os.path.join(
    BASE_DIR,
    f'../../data/results/excel/kpi_comparison_robustness_test_{today_date}.xlsx'
)

# Ensure the directory exists
kpi_excel_dir = os.path.dirname(kpi_excel_file_path)
os.makedirs(kpi_excel_dir, exist_ok=True)

# Save File
kpi_comparison.to_excel(kpi_excel_file_path)

# Adjust display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Adjust width to prevent wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column

print(kpi_comparison.iloc[:, 1:])
