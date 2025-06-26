# Load
import pandas as pd
import numpy as np

from k_means_constrained import KMeansConstrained
from src.data_prep.data_utils import compute_centroid


# load real data
machine_df = pd.read_csv("../../data/processed_data/machine_df.csv", index_col="Unnamed: 0")
werk_df = pd.read_csv("../../data/processed_data/cleaned_weighting_julich_2023.csv")

# load simulated field data
simulated_fields = pd.read_csv('../../data/simulated_data/simulated_fields.csv')

# =============
# Parameters
# =============
base_file_path = "../../data/"

# Regions
k = len(machine_df)  # Number of machines regions
max_fields_per_cluster = 600  # Maximum number of fields per machine (loader)
min_fields_per_cluster = 50   # Min...
total_fields = len(simulated_fields)  # Total number of fields

# Copy real machine_df to get df structure
simulated_machine_df = machine_df.copy()
#simulated_machine_df.drop(columns=['AccessibleCluster'], inplace=True)

# Set simulation column to nan to track changes
simulated_machine_df[['AccessibleFields', 'NumberOfFields', 'TotalYield', 'Route', 'DepotLocation']] = np.nan

print("Initialise simulated_machine_df", simulated_machine_df.head(2))

simulated_fields = simulated_fields[simulated_fields["Schlag-ID"] != 0]

# =========================
# Assign Regions
# =========================

# Validate constraints feasibility
if total_fields > k * max_fields_per_cluster:
    raise ValueError("Total number of fields exceeds the combined maximum capacity of all machines.")

# Perform Constrained K-means Clustering

# Extract the coordinates for clustering
coordinates = simulated_fields[['X', 'Y']].values

# Initialize the constrained K-means model
kmeans_constrained = KMeansConstrained(
    n_clusters=k,
    size_min=min_fields_per_cluster,  # Minimum size per cluster; adjust if needed
    size_max=max_fields_per_cluster,
    random_state=42
)

# Fit the model and predict cluster assignments
simulated_fields['cluster'] = kmeans_constrained.fit_predict(coordinates)


# Assign Region Clusters to Machines


# Create lists of "Schlag-ID" for each cluster
cluster_field_ids = {}
for cluster_num in range(k):
    field_ids = simulated_fields[simulated_fields['cluster'] == cluster_num]['Schlag-ID'].tolist()
    cluster_field_ids[cluster_num] = field_ids
    # Optionally, create variables like cluster_1, cluster_2, etc.
    globals()[f'cluster_{cluster_num+1}'] = field_ids

# Initialize the 'AccessibleFields' column as an object dtype to hold lists
if 'AccessibleFields' not in simulated_machine_df.columns:
    simulated_machine_df['AccessibleFields'] = pd.Series([[] for _ in range(len(simulated_machine_df))], dtype=object)
else:
    simulated_machine_df['AccessibleFields'] = simulated_machine_df['AccessibleFields'].astype(object)

# Assign each cluster to a machine
for idx, cluster_num in enumerate(range(k)):
    machine_number = simulated_machine_df.at[idx, 'Maus Nr.']
    simulated_machine_df.at[idx, 'AccessibleFields'] = cluster_field_ids.get(cluster_num, [])

# Verify Assignments

# Optionally, print the number of fields per machine to verify constraints
for idx, row in simulated_machine_df.iterrows():
    num_fields = len(row['AccessibleFields'])
    print(f"Machine {row['Maus Nr.']} has {num_fields} fields assigned.")

# Create a mapping from region cluster / region number to 'Maus Nr.'
cluster_to_maus_nr = {}
for idx in simulated_machine_df.index:
    machine_number = simulated_machine_df.at[idx, 'Maus Nr.']
    cluster_num = idx % k  # This should match the assignment logic above
    cluster_to_maus_nr[cluster_num] = machine_number

# Assign 'Maus Nr.' to each field in simulated_fields based on cluster
simulated_fields['Maus Nr.'] = simulated_fields['cluster'].map(cluster_to_maus_nr)

# Now, group simulated_fields by 'Maus Nr.' to compute the aggregates
agg_simulated_machine_df = simulated_fields.groupby('Maus Nr.').agg(
    AccessibleFields=('Schlag-ID', list),
    NumberOfFields=('Schlag-ID', 'count'),
    TotalYield=('reine Rüben (t)', 'sum')
).reset_index()

# Merge the aggregated data back into simulated_machine_df
simulated_machine_df = simulated_machine_df.merge(agg_simulated_machine_df, on='Maus Nr.', how='left')

# If you want to update other columns like 'NumberOfFields' and 'TotalYield' in simulated_machine_df
# and keep 'AccessibleFields' consistent, you can assign them directly
simulated_machine_df['NumberOfFields'] = simulated_machine_df['NumberOfFields_y']
simulated_machine_df['TotalYield'] = simulated_machine_df['TotalYield_y']
simulated_machine_df['AccessibleFields'] = simulated_machine_df['AccessibleFields_y']

# Drop the unnecessary columns created by the merge
simulated_machine_df = simulated_machine_df.drop(columns=['AccessibleFields_x', 'AccessibleFields_y',
                                                          'NumberOfFields_x', 'NumberOfFields_y',
                                                          'TotalYield_x', 'TotalYield_y'])


# Apply the centroid computation to each machine
simulated_machine_df['DepotLocation'] = simulated_machine_df['AccessibleFields'].apply(
    lambda fields: compute_centroid(fields, simulated_fields)
)

# Now, simulated_machine_df contains the updated information
# Adjust display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Adjust width to prevent wrapping
pd.set_option('display.max_colwidth', None)  # Show full content of each column

print("\nReal (selected)")
print(machine_df[["Maus Nr.", "NumberOfFields", "TotalYield"]])

print("\nSimulated")
print(simulated_machine_df)

# Save
simulated_machine_df.to_csv(f"{base_file_path}simulated_data/simulated_machine_df.csv")

print(f"\nSimulated Machine DF ist save to {base_file_path}simulated_data/simulated_machine_df.csv")