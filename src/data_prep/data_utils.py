# load
import pandas as pd
import numpy as np
import os
import pickle
import copy
import ast

from src.data_prep.data_generation import ArtificialDataGenerator, MachineDataGenerator
from src.data_prep.production_planning import ProductionPlanning
from src.data_prep.data_io import save_raw_data


def compute_centroid(accessible_fields, fields_df):
    """
    Computes the centroid (mean X and Y) of the given accessible fields.

    Parameters:
    - accessible_fields: List of 'Schlag-ID's accessible by the machine.
    - fields_df: DataFrame containing 'Schlag-ID', 'X', and 'Y'.

    Returns:
    - Tuple of (centroid_x, centroid_y) if accessible_fields is not empty.
    - None if accessible_fields is empty.
    """
    if not accessible_fields:
        return None
    # Filter the fields_df to include only the accessible fields
    subset = fields_df[fields_df['Schlag-ID'].isin(accessible_fields)]
    if subset.empty:
        return None
    # Compute the mean of X and Y
    centroid_x = subset['X'].mean()
    centroid_y = subset['Y'].mean()
    return centroid_x, centroid_y


# Define a function to calculate speed based on distance
def calculate_speed(distance):
    if distance > 5:
        return 17.5  # max speed for distances > 5 km
    elif distance >= 0.1:
        return 0.5 + (distance - 0.1) * (17.5 - 0.5) / (5 - 0.1)  # linear interpolation for 0.1 <= distance <= 5
    else:
        return 0.5  # fallback, just in case for very small distances


# Convert the distance matrix to a travel time matrix
def convert_distance_to_travel_time_matrix(distance_data):
    # Check if the input is a DataFrame or a NumPy array
    if isinstance(distance_data, pd.DataFrame):
        travel_time_data = distance_data.copy()
        for col in travel_time_data.columns:
            travel_time_data[col] = travel_time_data[col].apply(lambda d: min(d / calculate_speed(d), 1))

    elif isinstance(distance_data, np.ndarray):
        travel_time_data = distance_data.copy()
        for i in range(travel_time_data.shape[1]):
            travel_time_data[:, i] = np.vectorize(lambda d: min(d / calculate_speed(d), 1))(travel_time_data[:, i])

    else:
        raise TypeError("Input must be a pandas DataFrame or a numpy ndarray.")

    return travel_time_data


def build_distance_and_cost_matrices(machine_df, field_locations_df, t_p=2):
    """
    Builds distance and cost matrices for each machine based on their accessible fields and depot locations.

    Parameters:
    - machine_df (pd.DataFrame): DataFrame containing machine data with at least the following columns:
        - 'AccessibleFields': Column with string representations of lists of accessible field IDs.
        - 'DepotLocation': Column with string representations of tuples (X, Y) indicating depot coordinates.
        - 'Maus Nr.': Unique identifier for each machine (used as key in the output dictionaries).
        - 'TravelCost': Numerical value representing the cost per unit distance for the machine.
        - 'OperationsCost': Numerical value representing the operational cost at each field for the machine.
    - field_locations_df (pd.DataFrame): DataFrame containing field location data with the following columns:
        - 'Schlag-ID': Field IDs, including '0' which represents the depot.
        - 'X': X-coordinate of the field.
        - 'Y': Y-coordinate of the field.

    Returns:
    - distance_matrices (dict): Dictionary where each key is a machine ID from 'Maus Nr.' and each value is a pandas DataFrame
      representing the raw distance matrix for the accessible fields of that machine, including the depot.
    - cost_matrices (dict): Dictionary where each key is a machine ID from 'Maus Nr.' and each value is a pandas DataFrame
      representing the cost matrix (including travel and operational costs) for the accessible fields of that machine,
      including the depot.
    """
    distance_matrices = {}  # Dictionary to store raw distance matrices keyed by machine ID
    cost_matrices = {}  # Dictionary to store cost matrices keyed by machine ID
    travel_times = {}
    # Iterate over each machine in 'machine_df'
    for idx, machine in machine_df.iterrows():
        # Step 1: Process 'AccessibleFields' for the current machine
        accessible_fields = machine['AccessibleFields']

        # Check if the value is already a list, if it's a string, then parse it
        if isinstance(accessible_fields, str):
            try:
                accessible_fields = ast.literal_eval(accessible_fields)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Error parsing AccessibleFields for machine {machine['Maus Nr.']}: {e}")

        # Ensure accessible_fields is a list of integers
        if isinstance(accessible_fields, list):
            accessible_fields = [int(field) for field in accessible_fields]
        else:
            raise ValueError(
                f"AccessibleFields must be a list or string representation of a list for machine {machine['Maus Nr.']}")

        # Add '0' to the list if it's not already present (representing the depot)
        if 0 not in accessible_fields:
            accessible_fields = [0] + accessible_fields

        # Remove duplicates and sort the list of accessible field IDs
        accessible_fields = sorted(set(accessible_fields))

        # Step 2: Subset 'field_locations_df' to include only accessible fields
        accessible_field_locations = field_locations_df[
            field_locations_df['Schlag-ID'].isin(accessible_fields)
        ].copy()  # Use .copy() to avoid modifying the original DataFrame

        # Step 3: Update the depot location for 'Schlag-ID' == 0
        if 'DepotLocation' in machine and pd.notnull(machine['DepotLocation']):
            try:
                depot_location = ast.literal_eval(machine['DepotLocation'])
                if not (isinstance(depot_location, tuple) and len(depot_location) == 2):
                    raise ValueError
            except (ValueError, SyntaxError):
                raise ValueError(f"Invalid DepotLocation format for machine {machine['Maus Nr.']}")
        else:
            depot_location = (0, 0)  # Default depot location if not specified

        # Update the 'X' and 'Y' coordinates for the depot in 'accessible_field_locations'
        depot_mask = accessible_field_locations['Schlag-ID'] == 0
        if depot_mask.any():
            accessible_field_locations.loc[depot_mask, ['X', 'Y']] = depot_location
        else:
            # If depot is not in accessible_field_locations, append it
            depot_df = pd.DataFrame({
                'Schlag-ID': [0],
                'X': [depot_location[0]],
                'Y': [depot_location[1]]
            })
            accessible_field_locations = pd.concat([accessible_field_locations, depot_df], ignore_index=True)

        # Ensure the DataFrame is sorted by 'Schlag-ID' for consistent ordering
        accessible_field_locations.sort_values('Schlag-ID', inplace=True)
        accessible_field_locations.reset_index(drop=True, inplace=True)

        # Step 4: Calculate the raw distance matrix
        # Extract coordinates as a NumPy array
        coords = accessible_field_locations[['X', 'Y']].values

        # Compute the pairwise Euclidean distances between all accessible fields
        # This results in a square matrix where entry (i, j) is the distance between fields i and j
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2).astype(np.float32)

        # Step 5: Apply Travel and Operational Costs to the Distance Matrix

        # Multiply the distance matrix by the machine's 'TravelCost'
        travel_cost_matrix = dist_matrix * machine['TravelCost']

        travel_time_matrix = (convert_distance_to_travel_time_matrix(dist_matrix) / t_p)

        # TODO: Change if needed

        # including partial travel time cost
        travel_cost_travel_time_matrix = travel_cost_matrix + (1 - travel_time_matrix) * machine['OperationsCost'] * t_p

        # excluding partial travel time cost
        # travel_cost_travel_time_matrix = travel_cost_matrix

        # Set the diagonal elements to the machine's 'OperationsCost'
        # This represents the operational cost at each field
        np.fill_diagonal(travel_cost_travel_time_matrix, machine['OperationsCost'] * t_p)

        # Specifically set the depot's operational cost to 0
        # Find the index of the depot (Schlag-ID == 0)

        depot_index = accessible_fields.index(0) if 0 in accessible_fields else 0
        travel_cost_travel_time_matrix[accessible_fields.index(0), accessible_fields.index(0)] = 0

        # Step 6: Convert NumPy arrays to pandas DataFrames with proper indexing
        field_ids = accessible_fields  # Sorted list of accessible field IDs

        distance_df = pd.DataFrame(dist_matrix, index=field_ids, columns=field_ids)
        cost_df = pd.DataFrame(travel_cost_travel_time_matrix, index=field_ids, columns=field_ids)
        travel_time_df = pd.DataFrame(travel_time_matrix, index=field_ids, columns=field_ids)

        # Store the DataFrames in the respective dictionaries using the machine's ID as the key
        machine_id = machine['Maus Nr.']
        distance_matrices[machine_id] = distance_df
        cost_matrices[machine_id] = cost_df
        travel_times[machine_id] = travel_time_df

    # Return both dictionaries containing distance and cost DataFrames
    return distance_matrices, cost_matrices, travel_times


def raw_data_creation(scenarios, verbose=None, usage=None):
    """
    Takes number of fields as inputs and simulates a field network based on simuluation parameters (that are fixed).
    When testing smaller instances we are take a random sample of the original 300 set.

    In case parameters have to be changed, we need to go into the class itself or redevelop the code.
    """

    # Loop through each scenario
    for scenario in scenarios:
        scenario_id = scenario["scenario"]
        n_fields = scenario["n_fields"]
        max_radius = scenario["max_radius"]

        # Initialize data generators for the current scenario
        data_generator = ArtificialDataGenerator(n_fields, max_radius)
        data_generator.generate_field_locations(verbose=verbose, max_radius=max_radius)
        data_generator.calculate_distances_and_costs(cost_per_km=1)
        data_generator.generate_field_size()
        data_generator.generate_beet_yield()
        data_generator.initialize_sugar_concentration()

        if verbose:
            print(data_generator.field_locations)
            print(data_generator.cost_matrix)
            print(data_generator.field_size)
            print(data_generator.beet_yield)
            print(data_generator.sugar_concentration)

        # Save all generated data for the current scenario
        save_raw_data(scenario_id, data_generator)

        if usage:
            return (
                data_generator.field_locations,
                data_generator.cost_matrix,
                data_generator.field_size,
                data_generator.beet_yield,
                data_generator.sugar_concentration
            )

    print("All scenario data saved successfully.")


def instance_data_creation(instance_scenarios, field_df, distance_matrix_input,
                           loader_data_input=None, production_demand=13500, t_p=2,
                           base_file_path="../../data/", name="instance_data", usage=None):
    # Define the primary and alternative file paths
    alternative_path = "../data/"

    # Check if the primary path exists, use alternative if not
    if not os.path.exists(base_file_path):
        base_file_path = alternative_path

    data_store = {}  # Initialize a dictionary to store data for all scenarios

    for scenario in instance_scenarios:
        # Data Setup
        n_fields = int(scenario["nr_fields"])
        nr_h = int(scenario["nr_h"])
        nr_l = int(scenario["nr_l"])

        scenario_id = f"{n_fields}_{nr_h}_{nr_l}"

        field_locations = field_df[["Schlag-ID", "X", "Y"]].copy()

        distance_matrix = distance_matrix_input.copy()
        cost_matrix = distance_matrix.copy()
        np.fill_diagonal(distance_matrix[1:, 1:], 0)

        # Create field attribute dics
        field_size_dict = dict(zip(field_df['Schlag-ID'][1:], field_df['Erfasste Fläche (ha)'][1:]))
        beet_yield_dict = dict(zip(field_df['Schlag-ID'][1:], field_df["Rübenertrag (t/ha)"][1:]))
        sugar_concentration_dict = dict(zip(field_df['Schlag-ID'][1:], field_df['Zuckergehalt (%)'][1:]))
        beet_volume_dict = dict(zip(field_df['Schlag-ID'][1:], field_df['reine Rüben (t)'][1:]))

        # Machine Data Generator Class
        machine_data_generator = MachineDataGenerator(field_locations, n_fields, nr_h, nr_l)
        machine_data_generator.generate_machines()

        # Access the generated data
        harvester_data = machine_data_generator.harvesters
        if loader_data_input is None:
            loader_data = machine_data_generator.loaders
        else:
            loader_data = loader_data_input.copy()

        # Initialize c_t to be a copy of the distance matrix with diagonal set to 0
        c_t = cost_matrix * 0.1  # (10 cents per tonn per km)
        np.fill_diagonal(c_t, 0)

        # Initialize temporary matrices for a single time period and populate them

        try:
            h_distance_matrices, c_h, h_tt_matrices = build_distance_and_cost_matrices(harvester_data, field_locations,
                                                                                       t_p=t_p)
        except ValueError:
            print("Exception Loader Data used instead of Harvester Data")
            h_distance_matrices, c_h, h_tt_matrices = build_distance_and_cost_matrices(loader_data, field_locations,
                                                                                       t_p=t_p)
            harvester_data = loader_data.copy()

        l_distance_matrices, c_l, l_tt_matrices = build_distance_and_cost_matrices(loader_data, field_locations,
                                                                                   t_p=t_p)
        # Calculated assigned volume
        assigned_volume_per_machine = {}

        # Calculate the total volume assigned to each loader (machine), ignoring the depot (field 0)
        for row, fields in loader_data["AccessibleFields"].items():
            loader = loader_data["Maus Nr."].loc[row]

            # reset value
            value = 0

            # Check if the value is already a list, if it's a string, then parse it
            if isinstance(fields, str):
                fields = ast.literal_eval(fields)

            for field in fields:
                if field != 0:  # Ignore the depot
                    value += beet_volume_dict[field]

            assigned_volume_per_machine[loader] = value

        machine_ids = loader_data["Maus Nr."].values

        production_volume_per_day = production_demand
        total_volume = sum(list(assigned_volume_per_machine.values()))

        # Step 1: Initialize the ProductionPlanning class
        planning = ProductionPlanning(
            total_volume=total_volume,
            production_volume_per_day=production_volume_per_day,
            machine_ids=machine_ids,
            assigned_volume_per_machine=assigned_volume_per_machine,
            verbose=True
        )

        # Step 2: Calculate production plan with verbose output
        planning.calculate_production_plan()

        # Step 3: Redistribute the demand with verbose output
        start_date = '2023-09-18'  # Monday, 18th of September 2023

        # No Holidays
        # production_plan = planning.get_production_plan_without_holidays(start_date)

        # With holidays
        holidays = pd.to_datetime(['2023-10-03', '2023-12-24', '2023-12-25', '2023-12-26', '2024-01-01'])

        planning.redistribute_holiday_demand(start_date, holidays)

        planning.adjust_minimum_daily_demand(min_demand=1000)

        production_plan = planning.get_production_schedule()

        # Set sets
        Ih = {}
        Il = {}

        # Loader
        for loader_id in loader_data["Maus Nr."]:

            index = loader_data[loader_data["Maus Nr."] == loader_id].index[0]

            if isinstance(loader_data[loader_data["Maus Nr."] == loader_id].AccessibleFields[index], str):
                access_list = ast.literal_eval(
                    loader_data[loader_data["Maus Nr."] == loader_id].AccessibleFields[index])
            else:
                access_list = loader_data[loader_data["Maus Nr."] == loader_id].AccessibleFields[index]
            # Sort
            access = sorted(set(access_list))

            # Add 0
            access.insert(0, 0)
            Il[loader_id] = copy.deepcopy(access)

        # Assert I to be all values except for 0
        nodes = [item for sublist in Il.values() for item in sublist if item != 0]
        I = set(nodes)

        # Harvester
        for harvester_id, cost_matrix in c_h.items():
            rows, cols = cost_matrix.shape
            Ih[harvester_id] = range(rows)

        # Store all generated data for the current scenario in the dictionary
        data_store[scenario_id] = {
            "field_locations": field_locations,
            "distance_matrix": distance_matrix,
            "l_distance_matrices": l_distance_matrices,
            "l_tt_matrices": l_tt_matrices,
            "cost_matrix": cost_matrix,
            "sugar_concentration": sugar_concentration_dict,
            "field_size": field_size_dict,
            "beet_yield": beet_yield_dict,
            "beet_volume": beet_volume_dict,
            "I": I,
            "Ih": Ih,
            "Il": Il,
            "H": set(harvester_data["Maus Nr."]),
            "L": set(loader_data["Maus Nr."]),
            'c_t': c_t,
            'c_h': c_h,
            'c_l': c_l,
            'harvester_data': harvester_data,
            'loader_data': loader_data,
            "production_volume_per_day": production_volume_per_day,
            "assigned_volume_per_machine": assigned_volume_per_machine,
            'production_plan': production_plan
        }

    # Save the dictionary as a pickle file
    file_path = f"{base_file_path}/results/instances/{name}.pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(data_store, file)

    print(f" \n All instance scenarios saved to {file_path} \n")

    if usage:
        return data_store


def filter_instance_data(instance_data, loader_ids, harvester_ids, nr_fields_list=None, exclude_holidays=False,
                         reschedule=False):
    """
    Filters the instance_data based on specified loader_ids, harvester_ids, and an optional list of nr_fields.

    Parameters:
    - instance_data (dict): The original instance data loaded from the pickle file.
    - loader_ids (list): List of loader IDs to include.
    - harvester_ids (list): List of harvester IDs to include.
    - nr_fields_list (list, optional): List of numbers of fields to include. If not provided, all fields will be used.
    - exclude_holidays:
    - reschedule (bool): Reschedule is used in single experiments to align demands

    Returns:
    - filtered_data_store (dict): The filtered instance data.
    """
    filtered_data_store = {}

    for scenario_id, data in instance_data.items():

        # If nr_fields_list is None, create a single element list with None to use full fields
        nr_fields_list = nr_fields_list if nr_fields_list is not None else [None]

        # Iterate over nr_fields_list if provided, else handle full field set
        for nr_fields in nr_fields_list:

            # Filter loader_data
            filtered_loader_data = data['loader_data'].loc[
                data['loader_data']['Maus Nr.'].isin(loader_ids)
            ].copy()

            # If no loaders match, skip this scenario
            if filtered_loader_data.empty:
                print(f"Scenario {scenario_id}: No matching loaders found. Skipping.")
                continue

            # Filter harvester_data
            filtered_harvester_data = data['harvester_data'].loc[
                data['harvester_data']['Maus Nr.'].isin(harvester_ids)
            ].copy()

            # If no harvesters match, skip this scenario
            if filtered_harvester_data.empty:
                print(f"Scenario {scenario_id}: No matching harvesters found. Skipping.")
                print(f"Harvesters_df:\n {data['harvester_data']}")
                continue

            # Update 'H' and 'L'
            H = set(harvester_ids)
            L = set(loader_ids)

            # Update 'c_h' and 'c_l' based on filtered data
            c_h = {k: v for k, v in data['c_h'].items() if k in harvester_ids}
            c_l = {k: v for k, v in data['c_l'].items() if k in loader_ids}

            Il = {}
            Ih = {}

            # Update 'I' and 'Il' for loaders
            for loader_id in loader_ids:
                # Get the original list of fields assigned to this loader
                original_Il = data['Il'][loader_id]

                # Keep full field list if nr_fields is None, otherwise limit to nr_fields
                if nr_fields is None:
                    Il[loader_id] = sorted(original_Il)
                else:
                    Il[loader_id] = sorted(original_Il)[:nr_fields + 1]

                # Ensure 0 is included and not double-counted
                if 0 not in original_Il:
                    original_Il = [0] + original_Il

            # Assert I to be all values except for 0
            nodes = [item for sublist in Il.values() for item in sublist if item != 0]
            I = set(nodes)

            # Update loader_data

            for index, row in filtered_loader_data.iterrows():
                machine_id = row["Maus Nr."]  # Assuming "Maus Nr." is the Schlag-ID equivalent

                if machine_id in Il.keys():
                    # Update AccessibleFields
                    filtered_loader_data.at[index, "AccessibleFields"] = str(
                        [item for item in Il[machine_id] if item != 0])
                    # Update NumberOfFields
                    filtered_loader_data.at[index, "NumberOfFields"] = len(Il[machine_id]) - 1

            # Update 'Ih' for harvesters
            for harvester_id in harvester_ids:
                # Get the original list of fields assigned to this harvester
                original_Ih = data['Ih'][harvester_id]

                # Keep full field list if nr_fields is None, otherwise limit to nr_fields
                if nr_fields is None:
                    Ih[harvester_id] = sorted(original_Ih)
                else:
                    sorted_fields = sorted([field_id for field_id in original_Ih if field_id != 0])
                    Ih[harvester_id] = [0] + sorted_fields[:nr_fields]

            # Update beet_yield, field_size, sugar_concentration, beet_volume
            beet_yield = {k: v for k, v in data['beet_yield'].items() if k in I}
            field_size = {k: v for k, v in data['field_size'].items() if k in I}
            sugar_concentration = {k: v for k, v in data['sugar_concentration'].items() if k in I}
            beet_volume = {k: v for k, v in data['beet_volume'].items() if k in I}

            # extract the relevant columns
            production_plan_full = data['production_plan'].copy()

            # Construct the list of columns to select
            loader_goal_columns = [f"{loader_id} Goal" for loader_id in loader_ids]

            # Check which of these columns exist in the production_plan_full
            existing_loader_goal_columns = [col for col in loader_goal_columns if col in production_plan_full.columns]

            # If none of the columns exist, warn or skip
            if not existing_loader_goal_columns:
                print(
                    f"No matching loader goal columns found for loaders {loader_ids}. Skipping scenario {scenario_id}.")
                continue

            # Subset the production_plan
            production_plan = production_plan_full[['Date', 'Holiday'] + existing_loader_goal_columns].copy()

            # Align production goals with filtering out holidays effect (no effect if there are no holidays)
            # Make sure first week does not have multiple holidays
            production_volume_per_day = production_plan.loc[:6, existing_loader_goal_columns].sum().sum() * (1 / 7)
            loading_volume_per_day = production_plan.loc[:6, existing_loader_goal_columns].sum().sum() * (1 / 6)

            # Update 'assigned_volume_per_machine' for loaders
            assigned_volume_per_machine = {}
            for loader_id in loader_ids:
                volume = sum(beet_volume[field_id] for field_id in Il[loader_id] if field_id != 0)
                assigned_volume_per_machine[loader_id] = volume
            print("Assigned Volume per machine: ", assigned_volume_per_machine)
            # Update 'total_volume'
            total_volume = sum(assigned_volume_per_machine.values())

            # Reschedule is used in single experiments to align demands
            if reschedule:

                # Update production_volume_per_day with the daily sum of loading goals of each loader on the first day
                production_volume_per_day = production_plan.loc[0, existing_loader_goal_columns].sum()

                # Initialize the ProductionPlanning class with the filtered data
                planning = ProductionPlanning(
                    total_volume=total_volume,
                    production_volume_per_day=production_volume_per_day,
                    machine_ids=loader_ids,
                    assigned_volume_per_machine=assigned_volume_per_machine,
                    verbose=True
                )

                # Step 2: Calculate production plan with verbose output
                planning.calculate_production_plan()

                # Step 3: Redistribute the demand with verbose output
                start_date = '2023-09-18'  # Monday, 18th of September 2023

                if exclude_holidays:
                    # No Holidays
                    print("Production plan with rescheduling (instance_data) and no holidays")
                    production_plan = planning.get_production_plan_without_holidays(start_date)

                else:
                    # 1. Define Holidays
                    holidays = pd.to_datetime(['2023-10-03', '2023-12-24', '2023-12-25', '2023-12-26', '2024-01-01'])
                    # 2. Redistribute Holiday Demand
                    planning.redistribute_holiday_demand(start_date, holidays)
                    # 3. Define min demand and redistribute
                    planning.adjust_minimum_daily_demand(min_demand=1000)
                    # 4. Get and display the final production schedule DataFrame
                    print("Production plan with rescheduling and holidays")
                    production_plan = planning.get_production_schedule()

            # Store all filtered and updated data for the current scenario
            scenario_key = f"{nr_fields}_{len(loader_ids)}_{len(harvester_ids)}" if nr_fields else f"all_{len(loader_ids)}_{len(harvester_ids)}"

            filtered_data_store[scenario_key] = {
                "field_locations": data["field_locations"],
                "l_distance_matrices": {loader_id: data["l_distance_matrices"][loader_id].copy() for loader_id in
                                        loader_ids},
                "l_tt_matrices": {loader_id: data["l_tt_matrices"][loader_id].copy() for loader_id in loader_ids},
                "sugar_concentration": sugar_concentration,
                "field_size": field_size,
                "beet_yield": beet_yield,
                "beet_volume": beet_volume,
                "I": I,
                "Ih": Ih,
                "Il": Il,
                "H": H,
                "L": L,
                'c_t': data['c_t'].copy(),
                'c_h': c_h,
                'c_l': c_l,
                'harvester_data': filtered_harvester_data,
                'loader_data': filtered_loader_data,
                "production_volume_per_day": production_volume_per_day,
                "loading_volume_per_day": loading_volume_per_day,
                "assigned_volume_per_machine": assigned_volume_per_machine,
                'production_plan': production_plan
            }

    return filtered_data_store


def create_production_demand_per_machine(beet_volume, machine_data, production_volume_per_day, start_date, holidays,
                                         verbose=False):
    total_volumes_per_machine = []
    assigned_volume_per_machine = []

    for machine_id, machine in machine_data.iterrows():
        accessible_fields = machine['AccessibleFields']
        total_volume = sum(beet_volume[field - 1] for field in accessible_fields if field > 0)
        total_volumes_per_machine.append(total_volume)
        assigned_volume_per_machine.append(total_volume)

    if verbose:
        print(f"Total Volumes per Machine: {total_volumes_per_machine}")

    planning = ProductionPlanning(total_volume=sum(total_volumes_per_machine),
                                  production_volume_per_day=production_volume_per_day,
                                  num_machines=len(total_volumes_per_machine),
                                  assigned_volume_per_machine=total_volumes_per_machine)

    planning.calculate_production_plan(verbose=verbose)
    planning.redistribute_demand(start_date=start_date, holidays=holidays)

    production_schedule = planning.get_production_schedule()

    machine_ids = [f'Machine {i + 1}' for i in range(len(total_volumes_per_machine))]
    production_schedule = production_schedule.rename(
        columns={f'machine {i + 1} Goal': machine_ids[i] for i in range(len(machine_ids))})

    if verbose:
        print(production_schedule)

    return production_schedule


"""
# Example usage in combination with data_generation.py

field_locations_all, cost_matrix_all, field_size_all, beet_yield_all, sugar_concentration_all = \
    raw_data_creation([{"scenario": "30S", "n_fields": 30, "max_radius": 10}], verbose=False, usage=True)

instance_data = instance_data_creation([{"nr_fields": 30, "nr_h": 2, "nr_l": 2}], field_locations_all, cost_matrix_all,
                                       field_size_all, beet_yield_all, sugar_concentration_all, usage=True)

beet_volume = instance_data["30_2_2"]["beet_volume"]
machine_data = instance_data["30_2_2"]["loader_data"]
production_volume_per_day = 4000
start_date = "2024-09-02"
holidays = pd.to_datetime(['2024-09-15', '2024-09-23'])

production_schedule_df = create_production_demand_per_machine(beet_volume, machine_data, production_volume_per_day,
                                                              start_date, holidays, verbose=True)
"""
