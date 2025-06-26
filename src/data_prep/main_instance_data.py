import os
import warnings
import pandas as pd
import seaborn as sns
from src.data_prep.data_utils import instance_data_creation

warnings.filterwarnings('ignore')

# === Path Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/"))
SIM_DIR = os.path.join(DATA_DIR, "simulated_data")
FIGURE_DIR = os.path.join(DATA_DIR, "figures")

# === Load Data ===
field_df = pd.read_csv(os.path.join(SIM_DIR, "simulated_fields.csv"), index_col=0)
machine_df = pd.read_csv(os.path.join(SIM_DIR, "simulated_machine_df.csv"), index_col=0)
cost_matrix = pd.read_csv(os.path.join(SIM_DIR, "simulated_cost_matrix_df.csv"), index_col=0)

# === Parameters ===
n_machines = len(machine_df)
n_fields = machine_df.NumberOfFields.sum()

field_locations = field_df[["Schlag-ID", "X", "Y"]].copy()
cost_matrix_array = cost_matrix.to_numpy()

# === Instance Creation ===
sim_instance_data_2h = instance_data_creation(
    [{"nr_fields": n_fields, "nr_h": n_machines, "nr_l": n_machines}],
    field_df, cost_matrix_array, t_p=2,
    loader_data_input=machine_df, production_demand=13500,
    base_file_path=DATA_DIR, name="simulated_instance_data_2h", usage=True
)

sim_instance_data_1h = instance_data_creation(
    [{"nr_fields": n_fields, "nr_h": n_machines, "nr_l": n_machines}],
    field_df, cost_matrix_array, t_p=1,
    loader_data_input=machine_df, production_demand=13500,
    base_file_path=DATA_DIR, name="simulated_instance_data_1h", usage=True
)

"""
# === Real Data (Commented Out for Now) ===

field_df = pd.read_csv(os.path.join(DATA_DIR, "processed_data/raw_field_data_julich_2023.csv"), index_col=0)
field_df["Rübenertrag (t/ha)"] = field_df["Rübenertrag (t/ha)"].clip(lower=55)

machine_df = pd.read_csv(os.path.join(DATA_DIR, "processed_data/machine_df.csv"), index_col=0)
cost_matrix = pd.read_csv(os.path.join(DATA_DIR, "processed_data/cost_matrix_julich_2023_df.csv"), index_col=0)
cost_matrix_array = cost_matrix.to_numpy()

real_instance_data = instance_data_creation(
    [{"nr_fields": n_fields, "nr_h": n_machines, "nr_l": n_machines}],
    field_df, cost_matrix_array, t_p=2,
    loader_data_input=machine_df, production_demand=13500,
    base_file_path=DATA_DIR, name="real_instance_data_2h", usage=True
)
"""
