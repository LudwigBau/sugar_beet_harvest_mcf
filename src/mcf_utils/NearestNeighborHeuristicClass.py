import numpy as np
import ast
import math

class NearestNeighborHeuristic:
    """
    A simple heuristic class for planning routes for harvesters and loaders based on the nearest neighbor strategy.
    """

    def __init__(self, distance_matrix, harvester_data, loader_data, harvester_access, loader_access):
        """
        Initializes the SimpleHeuristic class with the necessary data.

        Parameters:
            distance_matrix (numpy.ndarray): The distances between fields.
            harvester_data (DataFrame): Configuration data for harvesters.
            loader_data (DataFrame): Configuration data for loaders.
            harvester_access (dict): Accessible fields for each harvester, given as string representations of lists.
            loader_access (dict): Accessible fields for each loader, given as string representations of lists.
        """
        self.distance_matrix = distance_matrix
        self.harvester_data = harvester_data
        self.loader_data = loader_data
        self.harvester_access = self.convert_access_string_to_list(harvester_access)
        self.loader_access = self.convert_access_string_to_list(loader_access)
        self.harvested_fields = set()  # Initialize as an empty set to track harvested fields
        self.routes = {}  # Initialize as an empty dictionary to store routes for each machine

    @staticmethod
    def convert_access_string_to_list(access_data):
        """
        Convert string representations of field access lists into actual lists of integers.
        If the value is already a list or number, return it as is.

        Parameters:
        - access_data: Dictionary with keys as field names and values as either strings of lists or lists of integers.

        Returns:
        - Dictionary with keys as field names and values as lists of integers or the original value.
        """
        converted_access = {}
        for key, value in access_data.items():
            if isinstance(value, str):
                try:
                    # Safely evaluate the string to convert it to a list of integers
                    evaluated_value = ast.literal_eval(value)
                    if isinstance(evaluated_value, list):
                        converted_access[key] = evaluated_value
                    else:
                        # If the evaluated value is not a list, return the original value
                        converted_access[key] = value
                except (ValueError, SyntaxError):
                    # If evaluation fails, return the original value
                    converted_access[key] = value
            else:
                # If the value is not a string, return it as is
                converted_access[key] = value
        return converted_access

    def find_nearest_neighbor(self, current_index, accessible_fields, visited):
        """
        Finds the nearest unvisited neighbor from accessible fields.
        """
        distances = self.distance_matrix[current_index]
        min_distance = float('inf')
        next_index = None
        for i in accessible_fields:
            if not visited[i] and distances[i] < min_distance:
                min_distance = distances[i]
                next_index = i
        return next_index

    def plan_harvester_route(self, harvester_id, finish=0):
        """
        Plans the route for a harvester considering its accessible fields.
        """
        accessible_fields = self.harvester_access[harvester_id]  # Include depot
        visited = [False] * len(self.distance_matrix)
        current_index = 0  # Start at the depot
        visited[current_index] = True
        route = [current_index]

        while True:
            next_index = self.find_nearest_neighbor(current_index, accessible_fields, visited)
            if next_index is None:
                break
            visited[next_index] = True
            route.append(next_index)
            self.harvested_fields.add(next_index)  # Mark the field as harvested
            current_index = next_index
        route.append(finish)
        self.routes[harvester_id] = route  # Store the route for this harvester

    def plan_loader_route(self, loader_id, finish=0):
        """
        Plans the route for a loader considering its accessible fields and dependency on harvested fields.
        """
        accessible_fields = self.loader_access[loader_id]   # Include depot
        visited = [False] * len(self.distance_matrix)
        current_index = 0  # Start at depot
        visited[current_index] = True
        route = [current_index]

        while True:
            # Filter accessible fields to include only those that have been harvested
            accessible_and_harvested = [field for field in accessible_fields if
                                        field in self.harvested_fields or field == 0]
            next_index = self.find_nearest_neighbor(current_index, accessible_and_harvested, visited)
            if next_index is None:
                break
            visited[next_index] = True
            route.append(next_index)
            current_index = next_index
        route.append(finish)
        self.routes[loader_id] = route

    def plan_all_routes(self):
        """
        Plans routes for all harvesters and loaders.
        """
        for harvester_id in self.harvester_access:
            self.plan_harvester_route(harvester_id)
        for loader_id in self.loader_access:
            self.plan_loader_route(loader_id)

    def get_routes(self):
        """
        Returns the planned routes.
        """
        return self.routes

