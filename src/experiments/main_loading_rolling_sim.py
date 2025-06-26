# Regular Libraries
import os
import pandas as pd
import pickle
from datetime import datetime

# Custom
from src.data_prep.data_utils import filter_instance_data
from src.mcf_utils.RealBenchRoutingClass import PLBenchScheduler
from src.mcf_utils.rolling_flow_utils import create_weekly_brackets, create_machine_groups
from src.mcf_utils.rolling_flow_utils import loader_rolling_flow_experiments_TimeAgg
from src.mcf_utils.rolling_results_utils import create_consolidated_excel_with_kpi_comparison

# Define the base file path for saving data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_file_path = os.path.join(BASE_DIR, "../../data/")

# Define Size / Load Data
# Gurobi Model Versions
gurobi_model_versions_2h = {"idle_time": True, "restricted": True, "access": False,
                            "working_hours": False, "travel_time": True, "t_p": 2}

gurobi_model_versions_1h = {"idle_time": True, "restricted": True, "access": False,
                            "working_hours": False, "travel_time": True, "t_p": 1}

# Gurobi Sensitivity Scenarios
sens = {"cost_multi": 1, "productivity_multi": 1, "working_hours_multi": 1, "LAMBDA": 0.5}

solver_params = {
    "TimeLimit": 3600,
    "MIPGap": 0.001,
    "ScaleFlag": 2,
    "Seed": 42,
    "DualReductions": 0,
    "Threads": 0
}

time_agg_params = {
    'vehicle_capacity_flag': True,
    'restricted_flow_flag1h': False,
    'restricted_flow_flag2h': False,
    'last_period_restricted_flag1h': True,     # Needs to be false
    'last_period_restricted_flag2h': True,      # We use more efficient restriction in coarse version
    'add_min_beet_restriction_flag1h': False,   # We like to use beet goals in fine version
    'add_min_beet_restriction_flag2h': False,   # Needs to be false
    'time_restriction': True,
    'verbose': False,
    'v_type': "binary"
}

# Load instance data, split and filter
print("Instance Data from: \n ../../data/results/instances/")

data_file_2h = os.path.join(BASE_DIR, '../../data/results/instances/simulated_instance_data_2h.pkl')
instance_data_2h = pd.read_pickle(data_file_2h)

data_file_1h = os.path.join(BASE_DIR, '../../data/results/instances/simulated_instance_data_1h.pkl')
instance_data_1h = pd.read_pickle(data_file_1h)

instance_keys = list(instance_data_1h.keys())
scenario = instance_keys[0]

# extract data
loaders = instance_data_1h[scenario]["L"]
production_plan = instance_data_1h[scenario]["production_plan"]

#
# Create groups
machine_groups = create_machine_groups(loaders, production_plan, group_ratio=2.5, verbose=True)
# select groups of interest
machine_groups = machine_groups[:].copy()

# Parameters
nr_fields = [600]  # per machine! [40], 600 for full test
harvester_ids = [1, 3]
n_clusters = 1
colors = ['blue', 'green', 'purple', 'grey']

filtered_instance_data_1h = {}
filtered_instance_data_2h = {}
weekly_results = {}

# Filter the instance_data per group
for group_id, group in enumerate(machine_groups):

    loader_ids = group

    print(f"\n== Start Group {group_id} ===\n")
    print(f"Containing Loader IDs: {group} \n")

    print("Start Filter Data 2h and 1h")

    # Filter the instance_data's
    filtered_instance_data_2h[group_id] = filter_instance_data(
        instance_data=instance_data_2h,
        loader_ids=loader_ids,
        harvester_ids=harvester_ids,
        nr_fields_list=nr_fields,
        exclude_holidays=False,
        reschedule=False
    )

    # Filter the instance_data
    filtered_instance_data_1h[group_id] = filter_instance_data(
        instance_data=instance_data_1h,
        loader_ids=loader_ids,
        harvester_ids=harvester_ids,
        nr_fields_list=nr_fields,
        exclude_holidays=False,
        reschedule=False
    )

    f_scenario = f"{nr_fields[0]}_{len(loader_ids)}_{len(harvester_ids)}"

    field_locations = filtered_instance_data_1h[group_id][f_scenario]["field_locations"]
    loader_data = filtered_instance_data_1h[group_id][f_scenario]["loader_data"]
    cost_matrix = filtered_instance_data_1h[group_id][f_scenario]["l_distance_matrices"]
    beet_yield = filtered_instance_data_1h[group_id][f_scenario]["beet_yield"]
    field_size = filtered_instance_data_1h[group_id][f_scenario]["field_size"]
    beet_volume = filtered_instance_data_1h[group_id][f_scenario]["beet_volume"]

    scheduler = PLBenchScheduler(field_locations, beet_yield, field_size, loader_data,
                                 n_clusters, colors, base_file_path, verbose=False, save_figures=False)

    # Route fields within regions and return routes
    region_routes = scheduler.route_fields_within_regions_and_visualize(plot=False)
    cut_routes = {}

    for loader, route in region_routes.items():
        cut_routes[loader] = route[:]

    # Output the routes per region
    for region, route in cut_routes.items():
        print(f"Region {region} route: {route}")

    total_volume = sum(filtered_instance_data_1h[group_id][f_scenario]["beet_volume"])
    production_volume_per_day = filtered_instance_data_1h[group_id][f_scenario]["production_volume_per_day"]
    production_plan = filtered_instance_data_1h[group_id][f_scenario]["production_plan"]

    #print("Production Plan 1h: ", production_plan)
    #print("Production_volume_per_day: ", production_volume_per_day)

    assigned_volume_per_machine = filtered_instance_data_1h[group_id][f_scenario]["assigned_volume_per_machine"]

    # **** 3. Split Sequence into Brackets ****
    weekly_brackets = create_weekly_brackets(region_routes, beet_volume, production_plan, verbose=False)

    # All splits, heuristic and optimization results are stored in a weekly result dict
    weekly_results[group_id] = loader_rolling_flow_experiments_TimeAgg(
        group_id=group_id,
        scenario_id=f_scenario,
        instance_raw_2h=filtered_instance_data_2h,
        instance_raw_1h=filtered_instance_data_1h,
        gurobi_model_versions_2h=gurobi_model_versions_2h,
        gurobi_model_versions_1h=gurobi_model_versions_1h,
        sens=sens,
        solver_params=solver_params,
        region_routes=cut_routes,
        base_file_path=base_file_path,
        max_iterations=6,
        params=time_agg_params
    )

    # Accessing results for the first week
    first_week = weekly_results[group_id]["Week_1"]
    print("First Week KPI Comparison:")
    print(first_week["kpi_comparison"])

# Results
sc = instance_data_1h[scenario]['sugar_concentration']
flow_type = "load_flow"
c_l = instance_data_1h[scenario]['c_l']

# Get today's date in yyyymmdd format
today_date = datetime.now().strftime('%Y%m%d')

# save weekly_results
weekly_results_file_path = os.path.join(BASE_DIR, f'../../data/results/reporting/weekly_sim_results_{today_date}.pkl')
# Save the dictionary as a pickle file
with open(weekly_results_file_path, 'wb') as file:
    pickle.dump(weekly_results, file)

# Update the file path with today's date
solution_file_path = os.path.join(BASE_DIR, f'../../data/results/reporting/weekly_sim_results_{today_date}.pkl')
results_file_path = os.path.join(BASE_DIR, "../../data/results/excel/")


for group_id, results in weekly_results.items():
    print("Start Results for Group: ", group_id)
    # Update the file path with today's date
    create_consolidated_excel_with_kpi_comparison(solution_file_path, results_file_path,
                                                  flow_type=flow_type,
                                                  group_id=group_id,
                                                  file_name="sim_data")
