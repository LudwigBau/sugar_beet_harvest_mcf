from simulation_utils import *
from src.data_prep.data_utils import instance_data_creation
from scipy.spatial.distance import pdist, squareform
from src.data_prep.data_generation import ArtificialDataGenerator
import pandas as pd
import numpy as np

# Parameters
nr_fields = 4133  # len(f2f_df)
distance = 60
#TODO: Adapt

min_sc = 5
max_sc = 25
min_size = 0.5  #
max_size = 50
min_yield = 70  # in dataset 20, here it is 70 after conduction expert interviews
max_yield = 140

# Load data
f2f_df = pd.read_csv("../../data/processed_data/cleaned_weighting_julich_2023.csv")
machine_df = pd.read_csv("../../data/processed_data/machine_df.csv")
location_data = pd.read_csv("../../data/processed_data/location_fields_julich_2023.csv")

# Extract data
sc_array = f2f_df['Zuckergehalt (%)'].values
size_array = f2f_df['Erfasste Fläche (ha)'].values
yield_array = f2f_df['Rübenertrag (t/ha)'].values

# Initialize Class
data_generator = ArtificialDataGenerator(nr_fields)

# Simulate fields locations
simulated_fields = data_generator.generate_field_locations_kde_clustered(location_data,
                                                                         grid_size=100,
                                                                         n_clusters_ratio=0.1,
                                                                         cluster_std=5,
                                                                         usage=True,
                                                                         verbose=False)

print("Simulated Fields", simulated_fields)

# Simulate Field Attributes
simulated_fields["Zuckergehalt (%)"] = data_generator.simulate_values(sc_array,
                                                                      min_val=min_sc,
                                                                      max_val=max_sc)

simulated_fields['Erfasste Fläche (ha)'] = data_generator.simulate_values(size_array,
                                                                          min_val=min_size,
                                                                          max_val=max_size)

simulated_fields['Rübenertrag (t/ha)'] = data_generator.simulate_values(yield_array,
                                                                        min_val=min_yield,
                                                                        max_val=max_yield)

simulated_fields['reine Rüben (t)'] = simulated_fields['Erfasste Fläche (ha)'] * simulated_fields['Rübenertrag (t/ha)']

# Define the new row with Jülich's coordinates
new_row = {
    'Schlag-ID': 0,
    'X': 383.908,            # Jülich's X coordinate divided by 1000
    'Y': 5641.227,           # Jülich's Y coordinate divided by 1000
    'Zuckergehalt (%)': np.nan,       # Set to NaN or a default value
    'Erfasste Fläche (ha)': np.nan,    # Set to NaN or a default value
    'Rübenertrag (t/ha)': np.nan,      # Set to NaN or a default value
    'reine Rüben (t)': np.nan          # Set to NaN or a default value
}

# Create a DataFrame for the new row
new_row_df = pd.DataFrame([new_row])

# Concatenate the new row with the existing DataFrame
simulated_fields = pd.concat([new_row_df, simulated_fields], ignore_index=True)

print(f"Simulated Field Data with {nr_fields} Fields: \n{simulated_fields}")

# Save Field data
simulated_fields.to_csv("../../data/simulated_data/simulated_fields.csv")


# Extract Distributions
sc_dist = fit_distribution(sc_array)
size_dist = fit_distribution(size_array)
yield_dist = fit_distribution(yield_array)

# Create a DataFrame to store the parameters
parameters = pd.DataFrame({
    'Parameter': ['nr_fields', 'distance', 'sc_dist', 'size_dist', 'yield_dist'],
    'Value': [nr_fields, distance, sc_dist[0], size_dist[0], yield_dist[0]]
})

print(f"Field Simulation Parameters: \n{parameters}")

# Save Parameters
parameters.to_csv("../../data/simulated_data/simulation_parameters.csv")


# Save Cost Martix:

# Step 1: Sort the DataFrame by 'Schlag-ID'
sorted_field_data = simulated_fields.sort_values(by='Schlag-ID').reset_index(drop=True)

# Step 2: Extract sorted Schlag-ID as a NumPy array
sorted_schlag_ids = sorted_field_data['Schlag-ID'].to_numpy(dtype=np.int32)

print("Sorted Schlag-ID Array:")
print(sorted_schlag_ids)

# Step 3: Extract X and Y coordinates as float32
coordinates = sorted_field_data[['X', 'Y']].to_numpy(dtype=np.float32)

# Step 4: Compute the pairwise Euclidean distances as float32
distance_vector = pdist(coordinates, metric='euclidean').astype(np.float32)

# Step 5: Convert the distance vector to a square matrix
distance_matrix = squareform(distance_vector).astype(np.float32)

# Step 6: Convert to a DataFrame with Schlag-ID as index and columns
distance_df = pd.DataFrame(distance_matrix, index=sorted_schlag_ids, columns=sorted_schlag_ids)

print("Distance Matrix:")
print(distance_df)

distance_df.to_csv("../../data/simulated_data/simulated_cost_matrix_df.csv")



