import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import Model, GRB, quicksum
from scipy.stats import gaussian_kde
from src.data_prep.simulation_utils import *
from scipy.spatial import distance_matrix
import warnings
import random


class ArtificialDataGenerator:
    def __init__(self, n_fields_, seed=42):
        """
        Initializes the data generator with the necessary parameters.

        Parameters:
        - n_fields_: Number of fields to generate.
        - seed: Random seed for reproducibility.
        """
        self.n_fields = n_fields_
        self.fields = range(1, self.n_fields + 1)  # Initialize fields range
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.field_locations = None
        self.cost_matrix = None
        self.simulated_actuals = None

    def generate_field_locations_kde_clustered(self, kde_data_df, grid_size=100, n_clusters_ratio=0.1, cluster_std=10,
                                               usage=False, verbose=False):
        """
        Generates field locations using KDE-based clustering.

        Parameters:
        - kde_data_df: DataFrame containing existing field locations with columns ['X', 'Y'].
        - grid_size: Resolution of the KDE grid.
        - n_clusters_ratio: Proportion of total fields to serve as cluster centers.
        - cluster_std: Standard deviation for cluster dispersion.
        - verbose: If True, visualizes the KDE and simulated locations.
        """
        # Set seeds for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Extract coordinates
        coords = kde_data_df[['X', 'Y']].values.T  # Transposed for scipy's gaussian_kde

        # Perform KDE using scipy
        kde = gaussian_kde(coords, bw_method='scott')  # 'scott' is a common bandwidth selector

        # Define the grid over which to evaluate KDE
        x_min, x_max = kde_data_df.X.min(), kde_data_df.X.max()
        y_min, y_max = kde_data_df.Y.min(), kde_data_df.Y.max()
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel()])

        # Evaluate KDE on the grid
        density = kde(grid_coords).reshape(X_grid.shape)

        # Normalize density to sum to 1 for probability distribution
        density_normalized = density / density.sum()

        # Determine number of clusters
        n_clusters = max(1, int(self.n_fields * n_clusters_ratio))  # Ensure at least one cluster

        # Flatten the density grid for sampling
        density_flat = density_normalized.ravel()

        # Sample cluster indices based on density probabilities
        cluster_indices = np.random.choice(
            np.arange(len(density_flat)),
            size=n_clusters,
            replace=True,
            p=density_flat
        )

        # Convert cluster indices back to coordinates
        cluster_x = X_grid.ravel()[cluster_indices]
        cluster_y = Y_grid.ravel()[cluster_indices]
        cluster_centers = np.column_stack((cluster_x, cluster_y))

        # Calculate points per cluster
        points_per_cluster = self.n_fields // n_clusters
        remainder = self.n_fields % n_clusters

        simulated_points = []

        for i, center in enumerate(cluster_centers):
            # Assign points per cluster, distributing any remainder
            n_points = points_per_cluster + (1 if i < remainder else 0)

            # Generate points around the cluster center
            points = np.random.normal(loc=center, scale=cluster_std, size=(n_points, 2))

            # Ensure points stay within study area boundaries
            points[:, 0] = np.clip(points[:, 0], x_min, x_max)
            points[:, 1] = np.clip(points[:, 1], y_min, y_max)

            simulated_points.append(points)

        # Concatenate all simulated points
        simulated_points = np.vstack(simulated_points)

        # Create GeoDataFrame for simulated points
        simulated_df = pd.DataFrame(simulated_points, columns=['X', 'Y'])
        simulated_df['Schlag-ID'] = range(1, self.n_fields + 1)
        simulated_geometry = [Point(xy) for xy in zip(simulated_df.X, simulated_df.Y)]
        self.field_locations = gpd.GeoDataFrame(simulated_df, geometry=simulated_geometry)

        if verbose:
            # Visualize KDE
            plt.figure(figsize=(10, 10))
            plt.contourf(X_grid, Y_grid, density, cmap='Reds')
            kde_data_df.plot(ax=plt.gca(), markersize=5, color='blue', alpha=0.5, label='Original Fields')
            self.field_locations.plot(ax=plt.gca(), markersize=5, color='green', alpha=0.5, label='Simulated Fields')
            plt.title("KDE-Based Clustered Field Locations")
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.legend()
            plt.show()

        if usage:
            return self.field_locations

    def generate_field_locations(self, max_radius, verbose=False, usage=False):
        """
       Generates field locations within a specified radius from the depot and optionally visualizes them.

       Parameters:
       - verbose: If True, plots the generated field locations.
       """

        np.random.seed(42)

        # Adjusted to generate n_fields instead of n_fields - 1
        angles = np.random.uniform(0, 2 * np.pi, self.n_fields)
        radii = np.sqrt(np.random.uniform(0, max_radius ** 2, self.n_fields))
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)

        # Including depot at 0,0 and adjusting field numbers to start from 1
        self.field_locations = pd.DataFrame(
            {'Schlag-ID': range(0, self.n_fields + 1), 'X': np.insert(x_coords, 0, 0), 'Y': np.insert(y_coords, 0, 0)})

        if verbose:
            plt.figure(figsize=(10, 10))
            plt.scatter(x_coords, y_coords, c='blue', label='Fields', s=10)
            plt.scatter([0], [0], c='red', label='Depot', s=50)  # Highlight depot
            plt.xlabel('Kilometers')
            plt.ylabel('Kilometers')
            plt.title('Field Locations including Depot')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()

        if usage:
            return self.field_locations

    def calculate_distances_and_costs(self, cost_per_km):
        """
        Calculates the distances between all pairs of fields (including the depot) and generates a dummy cost matrix
        based on these distances. Assumes the first entry in self.field_locations is the depot. Later machines can
        multiply their costs with this matrix
        """
        n = len(self.field_locations)  # Corrected to include depot correctly
        distances = np.zeros((n, n))

        # Assuming the first row/column in field_locations and distances is for the depot
        for i, loc_i in enumerate(self.field_locations.itertuples(index=False), start=0):
            for j, loc_j in enumerate(self.field_locations.itertuples(index=False), start=0):
                if i != j:
                    # No need to offset indices by 1
                    distances[i, j] = np.linalg.norm([loc_i.X - loc_j.X, loc_i.Y - loc_j.Y])

        # Calculate the cost matrix
        self.cost_matrix = distances * cost_per_km
        np.fill_diagonal(self.cost_matrix, 1)  # Adjusted: Cost for working in the field
        self.cost_matrix[0, 0] = 0  # Staying at the depot

    def simulate_values(self, actual_values, min_val=None, max_val=None):
        """
        Simulate Sugar Concentration on each field

        Parameters:
        - array of real sc values
        - min and max values to correct data
        """

        random.seed(42)

        # Initialize an array to store the sugar concentrations over time for each field
        num_fields = len(self.fields)

        actual_dist = fit_distribution(actual_values,
                                       dist_name=None,
                                       min_val=min_val,
                                       max_val=max_val)

        simulated_actuals = simulate_actuals(actual_values,
                                             dist_name=actual_dist[0],
                                             params=actual_dist[1],
                                             samples=num_fields,
                                             min_val=min_val,
                                             max_val=max_val,
                                             verbose=True)

        self.simulated_actuals = simulated_actuals

        return simulated_actuals

    def initialize_sugar_concentration(self, initial_min=13, initial_max=15, total_increase_min=1.5,
                                       total_increase_max=2.5, max_time=1440):
        """
        Initializes sugar concentration per t for each field with a random value within a specified range and simulates
        its increase over time with varying growth rates.

        Parameters:
        - initial_min: Minimum initial sugar concentration percentage.
        - initial_max: Maximum initial sugar concentration percentage.
        - total_increase_min: Minimum total increase in sugar concentration percentage over the simulation period.
        - total_increase_max: Maximum total increase in sugar concentration percentage over the simulation period.
        - max_time: Time until which the sugar concentration will increase (in hours).
        """

        random.seed(42)

        # Initialize an array to store the sugar concentrations over time for each field
        num_fields = len(self.fields)
        num_periods = max_time
        sugar_concentration = np.zeros((num_fields, num_periods))

        # Initialize sugar concentration for each field and simulate growth over time
        for i, field in enumerate(self.fields):
            initial_concentration = round(random.uniform(initial_min, initial_max), 1)
            total_increase = random.uniform(total_increase_min, total_increase_max)
            step_size = total_increase / max_time
            sugar_concentration[i, 0] = initial_concentration

            for t in range(1, num_periods):
                sugar_concentration[i, t] = min(sugar_concentration[i, t - 1] + step_size,
                                                initial_concentration + total_increase)

        self.sugar_concentration = sugar_concentration

    def generate_beet_yield(self, initial_min=60, initial_max=80):
        """
        Initializes sugar beet yield per ha for each field with a random value within a specified range.

        Parameters:
        - initial_min: Minimum initial sugar beet yield.
        - initial_max: Maximum initial sugar beet yield.

        Source for assumption: "In Germany, the average yield in 2019 was around 70 tonnes per hectare" (365 farmnet)
        Source P&L: Between 60 and 80 depending on region

        Note that we want to let volume grow depending on the size of t (2,4,8 hours)
        """

        random.seed(42)

        # Initialize an array to store the initial beet yield
        initial_yields = np.zeros(len(self.fields))

        # Set the initial yield for each field
        for i, field in enumerate(self.fields):
            initial_yields[i] = int(random.uniform(initial_min, initial_max))

        self.beet_yield = initial_yields

    def generate_field_size(self, initial_min=20, initial_max=30):  #TODO: in real setting use 6-30
        """
        Initializes field sizes in ha with a random value within a specified range. Constant over time

        Parameters:
        - initial_min: Minimum initial sugar beet yield
        - initial_max: Maximum initial sugar beet yield

        Source for assumption:  "Average Farm Size (not field) 60.5 ha" (cleanenergywire)
                                "Fostering biodversity on a 25ha field" (Südzucker)

        P&L:                    300-2000t Loading per spot 450/75 = 6, 2000/75 = 26.6

        => take min 6, max: 30 and use uniform distribution
        """
        random.seed(42)

        # Initialize an array to store the initial beet yield
        initial_size = np.zeros(len(self.fields))

        # Set the initial yield for each field
        for i, field in enumerate(self.fields):
            initial_size[i] = int(random.uniform(initial_min, initial_max))

        self.field_size = initial_size


class MachineDataGenerator:
    def __init__(self, field_locations, n_fields, n_harvesters, n_loaders):
        """
        Initializes the machine data generator with field locations and the number of machines.

        Parameters:
        - field_locations: DataFrame containing field locations.
        - field_size: array containing field sizes in ha
        - beet_yield: array containing field yields in t per ha
        - n_fields: Total number of fields.
        - n_harvesters: Number of harvesters.
        - n_loaders: Number of loaders.
        """
        self.field_locations = field_locations
        self.n_fields = n_fields
        self.n_harvesters = n_harvesters
        self.n_loaders = n_loaders

    def cutting_plane_assignment(self, n_machines):
        # Ensure field 0 (depot) is always included in the clustering
        field_positions = self.field_locations[['X', 'Y']].values

        # Convert Cartesian coordinates to polar coordinates
        center = np.mean(field_positions, axis=0)
        relative_positions = field_positions - center
        angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])

        # Sort fields by angle
        sorted_indices = np.argsort(angles)
        sorted_fields = [i for i in sorted_indices]

        # Create a list to hold the fields accessible to each machine
        accessible_fields = [[] for _ in range(n_machines)]

        # Assign fields to their respective angular slices
        fields_per_machine = len(sorted_fields) // n_machines
        for i in range(n_machines):
            accessible_fields[i] = sorted_fields[i * fields_per_machine:(i + 1) * fields_per_machine]

        # Ensure that every field is assigned
        all_fields = set(range(self.n_fields + 1))
        assigned_fields = set(field for fields in accessible_fields for field in fields)
        unassigned_fields = list(all_fields - assigned_fields)

        # Distribute unassigned fields to machines, ensuring coverage
        for i, field in enumerate(unassigned_fields):
            accessible_fields[i % n_machines].append(field)

        # Ensure each machine has access to the depot (field 0)
        for fields in accessible_fields:
            if 0 not in fields:
                fields.insert(0, 0)

        return accessible_fields

    def machine_field_volume_distance_assignment(self, n_machines):
        # Convert to dictionary for easy lookup
        field_locations = {row['Field']: (row['X'], row['Y']) for idx, row in self.field_locations.iterrows()}

        # Calculate the workload for each field including the depot (field 0)
        fields = {i: self.field_size[i - 1] * self.beet_yield[i - 1] if i != 0 else 0 for i in
                  range(0, self.n_fields + 1)}

        # Define machine locations (for simplicity, all at origin)
        machine_locations = {i: (0, 0) for i in range(n_machines)}

        # Function to calculate Euclidean distance
        def euclidean_distance(loc1, loc2):
            return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

        # Create a new model
        m = Model("FieldAssignmentWithDistance")

        # Suppress Gurobi output
        m.setParam('OutputFlag', 0)

        MIPGap = 0.005  # in fraction to represent %
        m.setParam(GRB.Param.MIPGap, MIPGap)

        # Decision variables
        x = m.addVars(fields.keys(), range(n_machines), vtype=GRB.BINARY, name="assign")
        max_load = m.addVar(vtype=GRB.CONTINUOUS, name="max_load")
        total_distance = m.addVar(vtype=GRB.CONTINUOUS, name="total_distance")

        # Set objective: minimize the combination of max load and total distance
        alpha = 0.5  # weight for workload balancing
        beta = 0.5  # weight for distance minimization

        m.setObjective(alpha * max_load + beta * total_distance, GRB.MINIMIZE)

        # Constraints
        # Each field is assigned to exactly one machine
        for f in fields:
            m.addConstr(quicksum(x[f, i] for i in range(n_machines)) == 1)

        # Calculate load on each machine and ensure it's within the max load
        for i in range(n_machines):
            m.addConstr(quicksum(fields[f] * x[f, i] for f in fields) <= max_load)

        # Calculate total distance
        m.addConstr(
            total_distance == quicksum(
                x[f, i] * euclidean_distance(field_locations[f], machine_locations[i])
                for f in fields
                for i in range(n_machines)
            )
        )

        # Optimize model
        m.optimize()

        # Output results
        assignments = [[] for _ in range(n_machines)]
        for f in fields:
            for i in range(n_machines):
                if x[f, i].X > 0.5:  # x[f, i].X is the value of the variable x[f, i] in the solution
                    assignments[i].append(f)

        # Ensure each machine has access to the depot (field 0)
        for fields in assignments:
            if 0 not in fields:
                fields.insert(0, 0)

        self.accessible_fields = assignments
        return assignments

    def generate_machine_specs(self, base_productivity, base_travel_cost, base_operations_cost,
                               accessible_fields, working_hours):
        return [
            {
                'Maus Nr.': i,
                'Productivity': base_productivity + (i % 2) * 10, #+ (i % 2) * 10,  # Alternate productivity for variety
                'TravelCost': base_travel_cost + (i % 2) * 1,  # Slightly different travel costs
                'OperationsCost': base_operations_cost + (i % 2) * 1,  # Slightly different operational costs
                'AccessibleFields': fields,
                'WorkingHours': working_hours# - (i % 2) #* 2  # Alternate working hours for variety
            }
            for i, fields in enumerate(accessible_fields)
        ]

    def generate_machines(self):

        # Generate accessible fields for harvesters and loaders
        #accessible_fields_harvesters = self.machine_field_volume_distance_assignment(self.n_harvesters)

        accessible_fields_harvesters = self.cutting_plane_assignment(self.n_harvesters)

        #accessible_fields_loaders = self.machine_field_volume_distance_assignment(self.n_loaders)
        accessible_fields_loaders = self.cutting_plane_assignment(self.n_loaders)

        # Check if the assignment methods for harvesters and loaders are the same
        if type(accessible_fields_harvesters) != type(accessible_fields_loaders):
            raise ValueError("The assignment methods for harvesters and loaders must be the same.")

        # Generate specs for harvesters and loaders (specs are currently for 2h periods)
        self.harvester_specs = self.generate_machine_specs(120, 2, 100, accessible_fields_harvesters, 20)
        self.loader_specs = self.generate_machine_specs(130, 2.5, 100, accessible_fields_loaders, 20)

        # Generate data for harvesters and loaders
        self.harvesters = pd.DataFrame(self.harvester_specs)
        self.harvesters['Type'] = 'Harvester'
        self.loaders = pd.DataFrame(self.loader_specs)
        self.loaders['Type'] = 'Loader'