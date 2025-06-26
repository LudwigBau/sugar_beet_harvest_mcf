import pandas as pd

from datetime import datetime

from src.mcf_utils.rolling_results_utils import create_consolidated_excel_with_kpi_comparison

# Load instance data, split and filter
instance_data = pd.read_pickle("../../data/results/instances/simulated_instance_data.pkl")
instance_keys = list(instance_data.keys())
scenario = instance_keys[0]

# Set params
sc = instance_data[scenario]['sugar_concentration']
flow_type = "load_flow"
c_l = instance_data[scenario]['c_l']


# Get today's date in yyyymmdd format
today_date = datetime.now().strftime('%Y%m%d')

# Set path
path = "../../data/results/reporting/weekly_sim_results_20250614.pkl"

# Load results
weekly_results = pd.read_pickle(path)

print(weekly_results.keys())
print(weekly_results[0].keys())

for group_id, results in weekly_results.items():
    print("Start Results for Group: ", group_id)
    # Update the file path with today's date
    solution_file_path = path

    base_file_path = "../../data/"
    results_file_path = "../../data/results/excel/"

    create_consolidated_excel_with_kpi_comparison(solution_file_path,
                                                  results_file_path,
                                                  flow_type=flow_type,
                                                  gap_rows=4,
                                                  group_id=group_id)

