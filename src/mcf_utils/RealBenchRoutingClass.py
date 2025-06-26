import numpy as np
import ast

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from matplotlib.patches import Polygon

from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

import random

from src.data_prep.data_utils import instance_data_creation


class PLBenchScheduler:
    def __init__(self, field_locations, beet_yield, field_size, loader_data,
                 n_clusters, colors, base_file_path, verbose=False, save_figures=False):

        # Convert Object list to number list
        for key, fields in loader_data["AccessibleFields"].items():
            # Check if fields need to be converted using ast.literal_eval
            if isinstance(fields, str):
                # Use .at to assign single cell values to avoid pandas thinking it's assigning an iterable
                loader_data.at[key, "AccessibleFields"] = ast.literal_eval(fields)

            else:
                # No conversion needed, but still use .at to avoid issues
                loader_data.at[key, "AccessibleFields"] = fields

            if isinstance(loader_data.loc[key, "DepotLocation"], str):
                # Evaluate the string as a Python literal
                eval_result = ast.literal_eval(loader_data.loc[key, "DepotLocation"])
                # Assign the evaluated result (tuple) to the cell using .at
                loader_data.at[key, "DepotLocation"] = eval_result  # .at is used for single element assignment

        # Parameters and settings
        self.verbose = verbose
        self.save_figures = save_figures
        self.colors = colors
        self.base_file_path = base_file_path

        # Load Data
        self.field_locations = field_locations
        self.beet_yield = beet_yield
        self.field_size = field_size
        self.loader_data = loader_data

        # Size
        self.n_loaders = len(loader_data["AccessibleFields"])
        self.n_clusters = n_clusters

        # Attributes
        self.AccessibleFields = loader_data["AccessibleFields"]

        # Depot_id
        self.depot_id = 0


        # Empty
        self.connecting_fields = None

    def get_region_points_and_field_ids(self, loader_data, field_locations, fields, i):
        """
        Use if depot != factory.

        Retrieve depot location, create a temporary DataFrame with field locations including the depot,
        and return the region points and field IDs for the depot and fields.

        Parameters:
        loader_data (pd.DataFrame): DataFrame containing loader data with depot locations.
        field_locations (pd.DataFrame): DataFrame containing the field locations with 'Schlag-ID', 'X', and 'Y' columns.
        fields (list): List of fields to include in the region points.
        i (int): Current index for retrieving the loader's depot location.

        Returns:
        np.ndarray: Region points (coordinates) including depot and fields.
        np.ndarray: Field IDs including depot and fields.
        """

        # Retrieve loader's depot location
        row_index = loader_data.index[i]
        depot_location = loader_data.loc[row_index, "DepotLocation"]

        # Assign a unique Schlag-ID for the depot
        depot_id = self.depot_id

        # Append depot to accessible fields
        extended_fields = fields + [depot_id]

        # Create a temporary DataFrame including the depot
        temp_field_locations = field_locations[field_locations["Schlag-ID"] != 0].copy()

        # Add the depot entry to the temporary DataFrame
        depot_entry = pd.DataFrame({
            'Schlag-ID': depot_id,
            'X': [depot_location[0]],
            'Y': [depot_location[1]]
        })
        temp_field_locations = pd.concat([temp_field_locations, depot_entry], ignore_index=True)

        # Extract coordinates and field IDs including depot
        region_points = temp_field_locations[
            temp_field_locations['Schlag-ID'].isin(extended_fields)
        ][['X', 'Y']].values
        field_ids = temp_field_locations[
            temp_field_locations['Schlag-ID'].isin(extended_fields)
        ]['Schlag-ID'].values

        return region_points, field_ids

    def plot_loader_regions(self):
        """Plot regions for each loader."""
        if self.verbose:
            plt.figure(figsize=(8, 8))
            colors = self.colors

            for i, fields in enumerate(self.loader_data["AccessibleFields"]):
                region_points = self.field_locations[
                    self.field_locations['Schlag-ID'].isin(fields)][['X', 'Y']].values
                plt.scatter(region_points[:, 0], region_points[:, 1],
                            c=[colors[i]], label=f'Loader {i + 1} Region', s=10)
                centroid = np.mean(region_points, axis=0)
                plt.scatter(centroid[0], centroid[1],
                            c='red', edgecolors=colors[i], s=50, label=f'Loader {i + 1} Center')

                # Plot individual depot locations
                depot_location = self.loader_data.loc[self.loader_data.index[i], "DepotLocation"]
                plt.scatter(depot_location[0], depot_location[1],
                            c='blue', marker='D', s=100, label=f'Depot (Loader {i + 1})')

            plt.legend()
            plt.xlabel('X Coordinate (km)')
            plt.ylabel('Y Coordinate (km)')
            plt.title('Loader Regions')
            plt.grid(True)

        if self.save_figures:
            plt.savefig(f"{self.base_file_path}figures/PL_loader_regions_{self.n_loaders}.png")

        if self.verbose:
            plt.show()


    def get_fields_per_polygon(self, accessible_fields, field_locations, region_index, plot=False):
        """
        Cluster fields within each loader's region using K-Means, ensuring depots are included as accessible fields.

        Parameters:
        - accessible_fields (list of lists): Fields accessible to each loader.
        - field_locations (pd.DataFrame): DataFrame containing 'Schlag-ID', 'X', and 'Y' of fields.
        - plot (bool): Whether to plot the polygons.

        Returns:
        - polygons_with_fields (list of lists): Fields contained in each polygon.
        - polygons_with_points (list of np.ndarray): Coordinates of points in each polygon.
        - centroids (list of np.ndarray): Centroids of each polygon.
        """

        # Validate clustering parameters
        min_fields_required = self.n_clusters * 3
        total_fields = sum(len(fields) for fields in accessible_fields)
        if total_fields < min_fields_required and self.n_clusters > 1:
            raise ValueError(
                f"Number of clusters ({self.n_clusters}) is too high for the number of fields ({total_fields}). "
                f"Each cluster needs at least 3 fields. Consider reducing the number of clusters."
            )

        max_clusters_per_loader = max(len(fields) // self.n_loaders for fields in accessible_fields)
        if self.n_clusters > max_clusters_per_loader and self.n_clusters > 1:
            raise ValueError(
                f"Number of clusters ({self.n_clusters}) is too high for the given number of loaders ({self.n_loaders}). "
                f"Consider reducing the number of clusters or increasing the number of loaders."
            )

        if plot:
            plt.figure(figsize=(8, 8))
        colors = self.colors

        polygons_with_fields = []
        polygons_with_points = []
        centroids = []


        for i, fields in enumerate(accessible_fields):

            # Get depot Location
            row_index = self.loader_data.index[region_index]
            depot_location = self.loader_data.loc[row_index, "DepotLocation"]

            # insert custom depots and align region points and fields ids
            region_points, field_ids = self.get_region_points_and_field_ids(
                self.loader_data, field_locations, fields, region_index)

            """# Retrieve loader's depot location
            row_index = self.loader_data.index[i]
            depot_location = self.loader_data.loc[row_index, "DepotLocation"]

            # Assign a unique Schlag-ID for the depot
            depot_id = 0

            # Append depot to accessible fields
            extended_fields = fields + [depot_id]

            # Create a temporary DataFrame including the depot
            temp_field_locations = field_locations[field_locations["Schlag-ID"] != 0].copy()

            depot_entry = pd.DataFrame({
                'Schlag-ID': depot_id,
                'X': [depot_location[0]],
                'Y': [depot_location[1]]
            })
            temp_field_locations = pd.concat([temp_field_locations, depot_entry], ignore_index=True)

            # Extract coordinates and field IDs including depot
            region_points = temp_field_locations[
                temp_field_locations['Schlag-ID'].isin(extended_fields)
            ][['X', 'Y']].values
            field_ids = temp_field_locations[
                temp_field_locations['Schlag-ID'].isin(extended_fields)
            ]['Schlag-ID'].values"""

            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10).fit(region_points)
            labels = kmeans.labels_

            # Identify the cluster containing the depot
            depot_index = len(region_points) - 1  # Depot is the last entry
            depot_cluster_label = labels[depot_index]

            # Swap depot's cluster label to 0 if it's not already
            if depot_cluster_label != 0:
                labels = np.where(labels == 0, -1, labels)  # Temporarily set label 0 to -1
                labels = np.where(labels == depot_cluster_label, 0, labels)  # Set depot's cluster to 0
                labels = np.where(labels == -1, depot_cluster_label, labels)  # Restore original label 0

            for cluster_idx in range(self.n_clusters):
                cluster_points = region_points[labels == cluster_idx]
                cluster_field_ids = field_ids[labels == cluster_idx]

                if len(cluster_points) > 2:
                    hull = ConvexHull(cluster_points)
                    polygon_points = cluster_points[hull.vertices]

                    polygon = Polygon(polygon_points, closed=True, facecolor=colors[i % len(colors)], alpha=0.4)

                    if plot:
                        plt.gca().add_patch(polygon)

                    polygons_with_fields.append(cluster_field_ids.tolist())
                    polygons_with_points.append(cluster_points)

                    # Calculate and store the centroid of the polygon
                    centroid = np.mean(polygon_points, axis=0)
                    centroids.append(centroid)

                elif len(cluster_points) == 2:
                    polygons_with_fields.append(cluster_field_ids.tolist())
                    polygons_with_points.append(cluster_points)
                    centroid = np.mean(polygon_points, axis=0)
                    centroids.append(centroid)

                elif len(cluster_points) == 1:
                    polygons_with_fields.append(cluster_field_ids)
                    polygons_with_points.append(cluster_points)
                    centroids.append(cluster_points)

            # Plot the region points and depot
            if plot:
                # Plot depot
                plt.scatter(depot_location[0], depot_location[1], c='red', edgecolors='red',
                            s=100, marker='D', label=f'Depot (Loader {i + 1})' if i == 0 else "")

                # Plot fields
                plt.scatter(region_points[:-1, 0], region_points[:-1, 1],
                            c=colors[i % len(colors)], label=f'Loader {i + 1} Region', s=10)

        if plot:
            # Handle legend to avoid duplicate depot labels
            handles, labels_plot = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels_plot, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.xlabel('X Coordinate (km)')
            plt.ylabel('Y Coordinate (km)')
            plt.title('Clustered Polygons in Loader Regions')
            plt.grid(True)

        if plot and self.save_figures:
            plt.savefig(f"{self.base_file_path}figures/PL_polygons_in_loader_regions_{self.n_loaders}_{self.n_clusters}.png")

        return polygons_with_fields, polygons_with_points, centroids

    def simple_greedy_route(self, coordinates, start_index=None, end_index=None):
        """Simple greedy route algorithm."""
        if start_index is None:
            start_index = 0

        route = [start_index]
        remaining_indices = list(range(len(coordinates)))
        remaining_indices.remove(start_index)

        while remaining_indices:
            current_index = route[-1]
            nearest_index = min(remaining_indices,
                                key=lambda i: np.linalg.norm(coordinates[current_index] - coordinates[i]))
            route.append(nearest_index)
            remaining_indices.remove(nearest_index)

        if end_index is not None:
            if route[-1] != end_index:
                if end_index in route:
                    route.remove(end_index)  # Remove end_index from its current position if already in route
                route.append(end_index)  # Ensure end_index is the last point

        return route

    def simulated_annealing(self, coordinates, initial_route, temperature=1000,
                            cooling_rate=0.995, stopping_temperature=1e-8, seed=42):
        """Simulated Annealing algorithm to optimize the order of visiting coordinates."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)  # If you use any NumPy random functions elsewhere

        def route_length(route):
            return np.sum(
                [np.linalg.norm(coordinates[route[i]] - coordinates[route[i + 1]]) for i in range(len(route) - 1)])

        def swap_two_points(route):
            new_route = route[:]
            if len(route) > 3:  # Ensure there are enough points to swap
                i, j = random.sample(range(1, len(route) - 1), 2)
                new_route[i], new_route[j] = new_route[j], new_route[i]
            return new_route

        current_route = initial_route[:]
        best_route = current_route[:]
        current_length = route_length(current_route)
        best_length = current_length

        while temperature > stopping_temperature:
            new_route = swap_two_points(current_route)
            new_length = route_length(new_route)

            if new_length < current_length or random.random() < np.exp((current_length - new_length) / temperature):
                current_route = new_route
                current_length = new_length

                if new_length < best_length:
                    best_route = new_route
                    best_length = new_length

            temperature *= cooling_rate

        return best_route

    def find_nearest_field_between_polygons(self, polygon1, polygon2, used_fields):
        """Find the nearest fields between two polygons, ensuring uniqueness."""
        min_distance = float('inf')
        nearest_pair = None

        # Iterate over all combinations of points from polygon1 and polygon2
        for point1 in polygon1:
            for point2 in polygon2:
                if tuple(point1) not in used_fields and tuple(point2) not in used_fields:
                    distance = np.linalg.norm(point1 - point2)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_pair = (point1, point2)

        # Mark these fields as used
        used_fields.add(tuple(nearest_pair[0]))
        used_fields.add(tuple(nearest_pair[1]))

        return nearest_pair

    def route_and_find_connecting_fields(self, field_locations, plot=False):
        """Route centroids and find connecting fields within the same loader's region."""
        if self.verbose:
            plt.figure(figsize=(8, 8))

        colors = self.colors
        all_connecting_fields = {}  # Dictionary to store connecting fields by region index

        for region_index, fields in enumerate(self.loader_data["AccessibleFields"]):

            # Get depot Location
            row_index = self.loader_data.index[region_index]
            depot_location = self.loader_data.loc[row_index, "DepotLocation"]

            # insert custom depots and align region points and fields ids
            region_points, field_ids = self.get_region_points_and_field_ids(
                self.loader_data, field_locations, fields, region_index)

            """region_points = self.field_locations[
                self.field_locations['Schlag-ID'].isin(fields)][['X', 'Y']].values"""

            ## FIT

            # Cluster the fields into polygons within each loader's region
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init=10).fit(region_points)
            labels = kmeans.labels_


            # Find the cluster that contains the point closest to the depot

            distances_to_depot = cdist([depot_location], region_points)
            closest_point_index = np.argmin(distances_to_depot)
            depot_cluster_label = labels[closest_point_index]


            # If the depot's cluster label isn't 0, swap it with the current label 0
            if depot_cluster_label != 0:
                labels[labels == 0] = -1  # Temporarily set the current 0 labels to -1
                labels[labels == depot_cluster_label] = 0  # Set the depot cluster labels to 0
                labels[labels == -1] = depot_cluster_label  # Set the former 0 labels to the depot's cluster label

            centroids = []
            polygons = []
            for cluster_idx in range(self.n_clusters):
                cluster_points = region_points[labels == cluster_idx]

                if len(cluster_points) > 2:  # Convex hull requires at least 3 points
                    hull = ConvexHull(cluster_points)
                    polygon_points = cluster_points[hull.vertices]
                    polygons.append(polygon_points)

                    # Calculate the centroid of the polygon
                    centroid = np.mean(polygon_points, axis=0)
                    centroids.append(centroid)

                    # Plot the polygon
                    if self.verbose:
                        polygon = Polygon(polygon_points, closed=True, facecolor=colors[region_index % len(colors)], alpha=0.4)

                        plt.gca().add_patch(polygon)

                        # Mark the centroid
                        plt.scatter(centroid[0], centroid[1], c='black', s=50, marker='x')
                else:
                    raise ValueError(f"Cluster is too small to define a polygon. Polygon Cluster holds "
                                     f"{len(cluster_points)} fields")

            # Route the centroids within this region using a greedy algorithm or any chosen TSP solver
            region_connecting_fields = []
            if centroids:
                initial_route = self.simple_greedy_route(centroids)
                optimized_route = self.simulated_annealing(centroids, initial_route)

                # Initialize used_fields outside the loop
                used_fields = set()  # Moved this line outside the loop over j
                used_fields.add(tuple(depot_location))  #

                # Find connecting fields between consecutive polygons
                for j in range(len(optimized_route) - 1):
                    start_polygon = polygons[optimized_route[j]]
                    end_polygon = polygons[optimized_route[j + 1]]

                    # Find the nearest fields to connect the polygons
                    start_field, end_field = self.find_nearest_field_between_polygons(
                        start_polygon, end_polygon, used_fields)
                    region_connecting_fields.append((start_field, end_field))

                    if plot:
                        plt.plot([start_field[0], end_field[0]],
                                 [start_field[1], end_field[1]], 'r-', lw=2)

                        # Number the centroids in the route order
                        plt.text(start_field[0], start_field[1], f'{j + 1}a', fontsize=12, color='black')
                        plt.text(end_field[0], end_field[1], f'{j + 1}b', fontsize=12, color='black')

            # Store connecting fields for the current region
            all_connecting_fields[region_index] = region_connecting_fields

            if plot:
                # Plot the region points
                plt.scatter(region_points[:, 0], region_points[:, 1], c=[colors[region_index]],
                            label=f'Loader {region_index + 1} Region', s=10)

        if plot:
            for i, loader in enumerate(self.loader_data.itertuples()):
                depot_location = loader.DepotLocation
                plt.scatter(depot_location[0], depot_location[1],
                            c='red', s=50, label=f'Depot (Loaders' if i == 0 else "")

            plt.legend()
            plt.xlabel('X Coordinate (km)')
            plt.ylabel('Y Coordinate (km)')
            plt.title('Routed Polygon Centroids and Connecting Fields within Regions')
            plt.grid(True)
            plt.savefig(f"{self.base_file_path}figures/PL_polygons_connecting_fields_{self.n_loaders}_{self.n_clusters}.png")
            plt.show()

        return all_connecting_fields  # Return connecting fields organized by region

    def route_fields_within_regions_and_visualize(self, plot=False):

        region_routes = {}
        connecting_fields = self.route_and_find_connecting_fields(self.field_locations)

        # Loop over each loader region
        for region_index, fields_in_region in enumerate(self.loader_data["AccessibleFields"]):

            # Get Loader_id

            # Define depot location
            # Get the actual index label for row `i`
            row_index = self.loader_data.index[region_index]
            depot_location = self.loader_data.loc[row_index, "DepotLocation"]

            loader_id = self.loader_data.loc[row_index, "Maus Nr."]

            # Get the fields and polygons for the current region
            polygons_with_fields, polygons_with_points, _ = self.get_fields_per_polygon(
                [fields_in_region], self.field_locations, region_index, plot=False
            )

            region_fields = connecting_fields[region_index]

            region_route = []


            # Reorder polygons
            reordered_polygons_with_fields = []
            reordered_polygons_with_points = []

            # Polygon 0 should contain the depot location
            # Calculate the minimum distance from depot to each polygon

            #_distances = [cdist([depot_location], polygon).min() for polygon in polygons_with_points]

            depot_id = self.depot_id
            # Find the index of the polygon with the smallest minimum distance
            depot_index = next(idx for idx, sublist in enumerate(polygons_with_fields) if depot_id in sublist)

            reordered_polygons_with_fields.append(polygons_with_fields[depot_index])
            reordered_polygons_with_points.append(polygons_with_points[depot_index])

            # Remove the depot polygon from the original list to avoid duplication
            polygons_with_fields.pop(depot_index)
            polygons_with_points.pop(depot_index)

            # Now find the polygons that contain each connecting field
            for field_pair in region_fields:
                target_field = field_pair[1]
                for j, polygon_points in enumerate(polygons_with_points):
                    if np.any(np.all(polygon_points == target_field, axis=1)):
                        reordered_polygons_with_fields.append(polygons_with_fields[j])
                        reordered_polygons_with_points.append(polygons_with_points[j])
                        # Remove the polygon from the original lists
                        polygons_with_fields.pop(j)
                        polygons_with_points.pop(j)
                        break

            # Update polygons with reordered lists
            polygons_with_fields = reordered_polygons_with_fields
            polygons_with_points = reordered_polygons_with_points

            # Routing each polygon in this region using Simulated Annealing
            for i, polygon_fields in enumerate(polygons_with_fields):
                if i == 0:
                    # Set the depot as the start field for the first polygon
                    #start_index = np.argmin(cdist([depot_location], polygons_with_points[i]))
                    start_index = polygons_with_fields[depot_index].index(depot_id)
                    start_coord = polygons_with_points[depot_index][start_index]
                    if self.verbose:
                        print(f"Region: {region_index}, Polygon: {i}, START: {start_coord}")
                else:
                    # Use the end field from the previous polygon as the start field
                    start_match = np.where(np.all(polygons_with_points[i] == region_fields[i - 1][1], axis=1))
                        
                    if start_match[0].size > 0:
                        start_index = start_match[0].item()
                        start_coord = polygons_with_points[i][start_index]
                        if self.verbose:
                            print(f"Region: {region_index}, Polygon: {i}, START: {start_coord}")
                    else:
                        raise ValueError(f"Start index not found for region {region_index}, polygon {i}")

                if i == len(polygons_with_fields) - 1:
                    # No end field for the last polygon
                    end_index = None
                    end_coord = None
                    if self.verbose:
                        print(f"Region: {region_index}, Polygon: {i}, END: None")
                else:
                    # Use the start field of the next polygon as the end field
                    end_match = np.where(np.all(polygons_with_points[i] == region_fields[i][0], axis=1))
                    if end_match[0].size > 0:
                        end_index = end_match[0].item()
                        end_coord = polygons_with_points[i][end_index]
                        if self.verbose:
                            print(f"Region: {region_index}, Polygon: {i}, END: {end_coord}")
                    else:
                        raise ValueError(f"End index not found for region {region_index}, polygon {i}")

                # Get the coordinates for this polygon
                coordinates = polygons_with_points[i]

                # Create an initial greedy route
                initial_route = self.simple_greedy_route(coordinates, start_index, end_index)

                # Optimize the route using Simulated Annealing
                optimized_route = self.simulated_annealing(coordinates, initial_route)

                #print("Initial Route: ", initial_route)
                #print("Optimised Route: ", optimized_route, "\n")

                routed_fields = [polygon_fields[i] for i in optimized_route]
                #print("Reouted fields: ", routed_fields)
                region_route.extend(routed_fields)

            # Save the route for the current region in a dictionary with the region index as the key
            region_routes[loader_id] = region_route
            region_routes[loader_id].append(0)
            # Visualization for the current region
            if plot:
                self.visualize_region_route(region_route, self.field_locations, region_index)

        return region_routes

    def visualize_region_route(self, region_route, field_locations, region_number):
        """
        Visualizes the final routed sequence of fields for a specific region.
        """
        # Retrieve loader's depot location
        row_index = self.loader_data.index[region_number]

        depot_location = self.loader_data.loc[row_index, "DepotLocation"]

        print("DEPOT LOCATION - REGION NUMBER", depot_location, region_number)
        # Assign a unique Schlag-ID for the depot
        depot_id = 0

        # Create a temporary DataFrame including the depot
        temp_field_locations = field_locations[field_locations["Schlag-ID"] != 0].copy()

        # Add the depot entry to the temporary DataFrame
        depot_entry = pd.DataFrame({
            'Schlag-ID': depot_id,
            'X': [depot_location[0]],
            'Y': [depot_location[1]]
        })
        temp_field_locations = pd.concat([temp_field_locations, depot_entry], ignore_index=True)

        coordinates = temp_field_locations.set_index('Schlag-ID').loc[region_route][['X', 'Y']].values

        colors = self.colors
        if self.verbose:
            # Initialize the figure only once
            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot the region route with arrows
            for i in range(len(coordinates) - 1):
                ax.arrow(coordinates[i, 0], coordinates[i, 1],
                         coordinates[i + 1, 0] - coordinates[i, 0],
                         coordinates[i + 1, 1] - coordinates[i, 1],
                         head_width=0.1, length_includes_head=True, color=colors[region_number], lw=2)

            # Plot all points
            ax.scatter(coordinates[:, 0], coordinates[:, 1], color=colors[region_number], s=50, zorder=5)

            # Plot the depot with a larger size and higher z-order to ensure visibility
            # Plot each depot individually

            ax.scatter(depot_location[0], depot_location[1],
                       color='red', s=50, label=f'Depot (Loader {region_number})' if i == 0 else '', zorder=10)

            ax.set_xlabel('X Coordinate (km)')
            ax.set_ylabel('Y Coordinate (km)')
            ax.set_title(f'Region {region_number} Route Visualization with Arrows')
            ax.grid(True)
            ax.legend()

            # Save the figure before showing it
            if self.save_figures:
                fig.savefig(f"{self.base_file_path}figures/PLroute_per_region_{self.n_loaders}_{region_number}.png")

            # Only show and close the figure if verbose or plot is enabled
            if self.verbose:
                plt.show()

            # Close the figure to avoid displaying it again in case of multiple plots
            else:
                plt.close(fig)  # Ensure the figure is closed if not showing


# Test on real data
"""
# Load Data
field_df = pd.read_csv("../../data/simulated_data/simulated_fields.csv",index_col=0)
machine_df = pd.read_csv("../../data/simulated_data/simulated_machine_df.csv",index_col=0)
production_volume = pd.read_csv("../../data/processed_data/loading_rates_julich_2023.csv", index_col="Ladedatum")
cost_matrix = pd.read_csv("../../data/simulated_data/simulated_cost_matrix_df.csv", index_col=0)

# Subset
subset_df = machine_df[(machine_df["Maus Nr."] == 46) | (machine_df["Maus Nr."] == 71)].copy()
# Downsize problem
subset_df.loc[1, "AccessibleFields"] = "[724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 1004, 1005, 1006, 1007, " \
                                       "1009, 1010, 1011, 1012, 1013, 1284, 1286, 1287, 1288, 1289, 1290, 1291, 1292, " \
                                       "1293, 1454, 1456, 1458, 4079, 4080, 4081, 4082, 4083]"

subset_df.loc[6, "AccessibleFields"] = "[49, 135, 139, 142, 143, 145, 147, 150, 155, 160, 161, 173, 215, 216, 217, " \
                                       "220, 222, 223, 243, 245, 267, 269, 271, 295, 296, 300, 301, 317, 319, 368, " \
                                       "2031, 2032, 2033, 2073, 2096, 3395, 3396, 3397, 3398, 3399, 3401]"

# Parameters
n_machines = len(subset_df)
n_fields = subset_df.NumberOfFields.sum()
n_clusters = 3
base_file_path = '../../data/'
colors = sns.color_palette("hls", n_machines)

print(len(subset_df["AccessibleFields"]))

field_locations = field_df[["Schlag-ID", "X", "Y"]].copy()
cost_matrix_array = cost_matrix.to_numpy()
field_size = field_df['Erfasste Fläche (ha)'][1:].values  # ignore factory entry
beet_yield = field_df["Rübenertrag (t/ha)"][1:].values
sugar_concentration = field_df['Zuckergehalt (%)'][1:].values

instance_data = instance_data_creation([{"nr_fields": n_fields, "nr_h": n_machines, "nr_l": n_machines}],
                                       field_locations, cost_matrix_array, field_size, beet_yield, sugar_concentration,
                                       loader_data_input=subset_df,
                                       production_demand=13200,
                                       base_file_path=base_file_path,
                                       name="subset_instance_data",
                                       usage=True
                                      )

instance_data = pd.read_pickle("../../data/results/instances/subset_instance_data.pkl")

# START SCHEDULER
base_file_path = "../../data/"
# Set up scheduler
print(instance_data.keys())

scheduler = PLBenchScheduler(field_locations, beet_yield, field_size,
                             instance_data[f"{n_fields}_{n_machines}_{n_machines}"]["loader_data"],
                             n_clusters, colors, base_file_path, verbose=True, save_figures=False)

# Plot Region
#scheduler.plot_loader_regions()

# Step 1: assign regions and polygons
polygons_with_fields, polygons_with_points, centroids = scheduler.get_fields_per_polygon(
    scheduler.AccessibleFields, scheduler.field_locations, region_index=0, plot=True)  # Works

# Output the fields per polygon
for idx, polygon_fields in enumerate(polygons_with_fields):
    print(f"Polygon {idx + 1} contains fields: {polygon_fields}")
    print(f"Locations: {polygons_with_points[idx]}")

# Step 2: route polygons and find connecting fields
connecting_fields = scheduler.route_and_find_connecting_fields(field_locations=field_df, plot=True)


# Output the connecting fields
print("Connecting fields between polygons:")
print(connecting_fields)


# Step 3: Route fields within regions, visualize all regions in one figure, and return routes
region_routes = scheduler.route_fields_within_regions_and_visualize(plot=True)

# Output the routes per region
for region, route in region_routes.items():
    print(f"Region {region} route: {route}")

print(region_routes)

"""


# Example usage:
"""
from src.data_prep.data_utils import raw_data_creation, instance_data_creation
# Parameters
n_machines = 2
n_fields = 150
n_clusters = 3
radius = 80
base_file_path = '../../data/'
colors = ['blue', 'green', 'purple', 'grey']

# Generate Data: 
field_locations, cost_matrix, field_size, beet_yield, sugar_concentration = \
    raw_data_creation([{"scenario": "30S", "n_fields": n_fields, "max_radius": radius}], verbose=False, usage=True)

instance_data = instance_data_creation([{"nr_fields": n_fields, "nr_h": n_machines, "nr_l": n_machines}],
                                        field_locations, cost_matrix, field_size, beet_yield, 
                                        sugar_concentration, usage=True)

# Set up scheduler
scheduler = PLBenchScheduler(field_locations, cost_matrix, beet_yield, field_size,
                             instance_data[f"{n_fields}_{n_machines}_{n_machines}"]["loader_data"],
                             n_clusters, colors, base_file_path, verbose=True, save_figures=True)

# Plot Region
#scheduler.plot_loader_regions()

# Step 1: assign regions and polygons
polygons_with_fields, polygons_with_points, centroids = scheduler.get_fields_per_polygon(
    scheduler.AccessibleFields, scheduler.field_locations)  # Works

# Output the fields per polygon
for idx, polygon_fields in enumerate(polygons_with_fields):
    print(f"Polygon {idx + 1} contains fields: {polygon_fields}")

# Step 2: route polygons and find connecting fields
connecting_fields = scheduler.route_and_find_connecting_fields()

# Output the connecting fields
print("Connecting fields between polygons:")
print(connecting_fields)


# Step 3: Route fields within regions, visualize all regions in one figure, and return routes
region_routes = scheduler.route_fields_within_regions_and_visualize(plot=True)

# Output the routes per region
for region, route in region_routes.items():
    print(f"Region {region} route: {route}")

print(region_routes)
"""