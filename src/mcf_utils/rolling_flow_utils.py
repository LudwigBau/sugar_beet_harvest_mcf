import pandas as pd
import pickle
import numpy as np
import os
from copy import deepcopy
from datetime import datetime

from collections import defaultdict
from typing import TypeAlias, Set, Dict, List, Tuple, Any, Optional

import gurobipy as gp
from gurobipy import GRB, quicksum

# Load utils
from src.mcf_utils.loading_flow_utils import enforce_schedule_with_constraints_idle
from src.mcf_utils.loading_flow_utils import from_schedule_to_hot_start_idle
from src.mcf_utils.loading_flow_utils import production_plan_holiday_to_t
# New loader flow utils imports
from src.mcf_utils.loading_flow_utils import (
    prepare_loader_flow_data,
    solve_loader_flow,
    generate_valid_arcs_for_all_loaders,
    extract_last_period_beet_volume
)

# result utils
from src.mcf_utils.single_results_utils import beet_movement_info, extract_machinery_schedule_idle
from src.mcf_utils.rolling_results_utils import create_kpi_comparison

# Load Classes
from src.mcf_utils.BeetFlowClass import BeetFlow
from src.mcf_utils.LoaderFlowClass import LoaderFlow

from src.mcf_utils.heuristic_utils import create_production_plan_schedule_tau


def create_machine_groups(
        machines: List[int],
        production_plan: pd.DataFrame,
        group_ratio: float = 2.5,
        verbose: bool = False
) -> List[List[int]]:
    """
    Organizes machines into groups based on their weekly workload from the production plan.

    This function sorts machines by their total weekly loading goal in descending order
    and then uses a greedy alternating assignment to distribute them into a calculated
    number of groups. The aim is to balance the overall workload across groups.

    Args:
        machines (List[int]): A list of machine IDs.
        production_plan (pd.DataFrame): A DataFrame containing the production plan,
            expected to have columns like '{machine_id} Goal' for weekly workload.
        group_ratio (float): A factor used to determine the number of groups. The
            number of groups is calculated as `len(machines) / group_ratio`.
            Defaults to 2.5, meaning roughly 2-3 machines per group.
        verbose (bool): If True, prints additional information about the grouping process.
            Defaults to False.

    Returns:
        List[List[int]]: A list of lists, where each inner list represents a group
            and contains the IDs of the machines assigned to that group.
    """
    list_machine_workload = []

    for m in machines:
        # Get weekly workload
        weekly_workload = production_plan.loc[0:6, f"{m} Goal"].sum()
        # Store data in tuple
        machine_workload = (m, weekly_workload)
        # Append to list of machines
        list_machine_workload.append(machine_workload)

    # Calculate number of machines per group
    n_groups = int(len(machines) / group_ratio)

    if verbose:
        print("Number of Groups: ", n_groups, "\n")

    # Sort by workload (descending)
    machines_sorted = sorted(list_machine_workload, key=lambda x: x[1], reverse=True)

    # Greedy alternating split into n_groups
    groups = [[] for _ in range(n_groups)]

    # Append from highest to lowest iterating over each group list
    for i, machine in enumerate(machines_sorted):
        groups[i % n_groups].append(machine)

    # Extract only machine IDs from groups
    machine_ids_groups = [[machine[0] for machine in group] for group in groups]

    # Output groups
    if verbose:
        for i, group in enumerate(machine_ids_groups):
            print(f"Group {i + 1}: {group}")

    return machine_ids_groups


def create_weekly_brackets(
        region_routes: Dict[int, List[int]],
        beet_volumes: Dict[int, float],
        production_plan: pd.DataFrame,
        verbose: bool = False
) -> Dict[int, List[Tuple[int, List[int]]]]:
    """
    Allocates fields from long-term routes into weekly processing brackets based on
    loader-specific production goals.

    This function simulates the process of breaking down a continuous harvesting plan
    into manageable weekly segments. It considers each loader's route and its assigned
    daily production goals from the production plan to determine which fields are
    scheduled for completion within each week.

    Args:
        region_routes (Dict[int, List[int]]): A dictionary mapping loader IDs to their
            long-term route, represented as a list of field (location) IDs.
        beet_volumes (Dict[int, float]): A dictionary mapping field IDs to their
            current beet volume.
        production_plan (pd.DataFrame): A DataFrame containing the overall production
            plan, including 'Date' and columns like '{loader_id} Goal' for daily targets.
        verbose (bool): If True, prints detailed information about the allocation process.
            Defaults to False.

    Returns:
        Dict[int, List[Tuple[int, List[int]]]]: A nested dictionary where:
            - The outer keys are `loader_id`s.
            - The values are lists of tuples. Each tuple contains:
                - The `week_number` (int) starting from 0.
                - A list of `field_ids` (List[int]) allocated to that loader for that week.
            Depot (0) locations are added at the beginning and end of each weekly route.

    Raises:
        ValueError: If the `production_plan` does not contain the necessary machine goal columns
            for a given loader, or if a specific loader's goals are not found.
    """
    # Ensure the production plan is sorted by 'Date'
    production_plan = production_plan.sort_values('Date').reset_index(drop=True)

    # Use the first date in production_plan as the start date
    start_date = production_plan['Date'].iloc[0]

    # Group production plan into weekly brackets
    production_plan['Week'] = (production_plan['Date'] - start_date).dt.days // 7

    if verbose:
        print("\nUpdated Production Plan", production_plan)

    # Ensure that the production_plan has the necessary machine goal columns
    machine_goal_columns = [col for col in production_plan.columns if 'Goal' in col]
    if not machine_goal_columns:
        raise ValueError("Production plan must contain machine_id goal columns like '1 Goal', '2 Goal', etc.")

    # Group by week and sum the goals for each machine
    weekly_production_goals = production_plan.groupby('Week')[machine_goal_columns].sum()

    # Initialize output data structure to store the weekly allocation
    weekly_field_allocation = {}

    # Iterate through the region routes and beet volumes to allocate fields
    for region, route in region_routes.items():
        if verbose:
            print(f"\nProcessing Region {region}...")

        field_allocation = []
        weekly_allocation = []
        current_week = 0
        current_weekly_production = 0

        # Determine the machine goal column for this region
        machine_goal_column = f"{region} Goal"
        if machine_goal_column not in weekly_production_goals.columns:
            raise ValueError(f"Production plan does not contain goals for {machine_goal_column}.")

        # Get the weekly production goals for this machine
        machine_weekly_goals = (weekly_production_goals[machine_goal_column]) * 1  # adapt if buffer is needed

        # Iterate through the fields in the route
        for field in route:
            if field == 0:
                volume = 0
            else:
                volume = beet_volumes[field]

            # Add field to weekly allocation
            weekly_allocation.append(field)
            current_weekly_production += volume

            if verbose:
                print(
                    f"Field {field} (volume {volume}) added. Current week production: {current_weekly_production} / {machine_weekly_goals.get(current_week, 'N/A')}"
                )

            # Get the current week's goal
            if current_week in machine_weekly_goals.index:
                weekly_goal = machine_weekly_goals.loc[current_week]
            else:
                weekly_goal = None

            # Check if weekly goal is met
            if weekly_goal is not None and current_weekly_production >= weekly_goal:
                # Store the allocated fields for this week
                field_allocation.append((current_week, weekly_allocation.copy()))

                # Verbose: check if the goal is met
                if verbose:
                    print(
                        f"Week {current_week} goal met. Production: {current_weekly_production}, Required: {weekly_goal}"
                    )

                # Move to the next week
                current_week += 1
                current_weekly_production = 0
                weekly_allocation = []

            elif weekly_goal is None:
                if verbose:
                    print("No more production goals available. Stopping allocation.")
                break  # No more production goals available

        # Check if there are any remaining fields after full weeks
        if weekly_allocation:
            field_allocation.append((current_week, weekly_allocation.copy()))
            if verbose:
                print(
                    f"Week {current_week} (Partial week or overhang) assigned remaining fields. Production: {current_weekly_production}."
                )

        # Store the final field allocation for the region
        weekly_field_allocation[region] = field_allocation

        # Verbose: check if all fields were visited
        if verbose:
            allocated_fields = [field for week, fields in field_allocation for field in fields if field != 0]
            missing_fields = set(route) - set(allocated_fields)
            if missing_fields:
                print(f"Fields not allocated in Region {region}: {missing_fields}")
            else:
                print(f"All fields in Region {region} were successfully allocated.")

    # Add zeros at the beginning and end of each route
    for region, allocations in weekly_field_allocation.items():
        for idx, (week, fields) in enumerate(allocations):
            # Ensure zero at the beginning
            if fields[0] != 0:
                fields.insert(0, 0)
            # Ensure zero at the end
            if fields[-1] != 0:
                fields.append(0)
            # Update the allocation
            weekly_field_allocation[region][idx] = (week, fields)

    return weekly_field_allocation


def update_volumes_and_routes(
        decision_variables: Dict[str, Any],
        beet_volumes: Dict[int, float],
        region_routes: Dict[int, List[int]],
        production_plan: pd.DataFrame,
        load_inventory: bool = False
) -> Tuple[Dict[int, float], Dict[int, List[int]], pd.DataFrame, float]:
    """
    Updates the remaining beet volumes on fields, adjusts loader routes, and truncates
    the production plan based on the results of an optimization model.

    This function is crucial for enabling the rolling horizon approach by preparing
    the state for the subsequent optimization iteration.

    Args:
        decision_variables (Dict[str, Any]): A dictionary containing decision variables
            from a solved Gurobi model, specifically requiring 'beet_flow_b'
            which holds the remaining beet volumes.
        beet_volumes (Dict[int, float]): The current (initial for this iteration)
            beet volumes at each field, mapped by field ID.
        region_routes (Dict[int, List[int]]): A dictionary mapping loader IDs to their
            current route (list of field IDs).
        production_plan (pd.DataFrame): The overall production plan DataFrame.
        load_inventory (bool): If True, the inventory level at the last time period
            from the 'beet_flow_b' variables will be extracted and returned.
            Defaults to False.

    Returns:
        Tuple[Dict[int, float], Dict[int, List[int]], pd.DataFrame, float]: A tuple containing:
            - updated_volumes (Dict[int, float]): A dictionary of updated beet volumes
              for each field after the current optimization iteration.
            - updated_routes (Dict[int, List[int]]): A dictionary of loader routes
              with fields that have been completely harvested (zero yield) removed.
            - updated_production_plan (pd.DataFrame): The production plan with the
              days corresponding to the just-optimized period removed.
            - updated_inventory (float): The total inventory level at the end of the
              optimized period. Returns `np.nan` if `load_inventory` is False.
    """
    # Extract the fields (I) and time periods (T) from decision variables
    I = {i for i, _, _, _ in decision_variables['beet_flow_b'].keys()}
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    # Initialize a list to store the updated beet volumes
    updated_volumes = beet_volumes.copy()

    # Iterate over fields and time periods to update volumes
    for i in I:
        # Update the corresponding field's volume at the last period
        updated_volumes[i] = np.round(decision_variables["beet_flow_b"][(i, 0, 0, max(T))], 0)

    if load_inventory:
        # Use the final period inventory as new inventory level
        updated_inventory = sum(decision_variables["beet_flow_b"][(i, 1, 1, max(T))] for i in I)
        print("Updated Inventory Levels: ", updated_inventory)
    else:
        # Existing logic if no inventory update
        updated_inventory = np.nan

    # Now update the region routes by removing fields with zero yield
    updated_routes = {}

    for region, fields in region_routes.items():
        updated_fields = [field for field in fields if field != 0 if updated_volumes[field] > 0]
        if updated_fields:  # Only add regions with remaining fields
            updated_routes[region] = updated_fields

    # Update the production plan based on the remaining days
    if len(production_plan) <= 7:
        # Keep the first 7 days and drop any days beyond that
        updated_production_plan = production_plan.iloc[:7].reset_index(drop=True)
    else:
        # Drop the first 7 days and keep the rest
        updated_production_plan = production_plan.iloc[7:].reset_index(drop=True)

    return updated_volumes, updated_routes, updated_production_plan, updated_inventory


def run_rolling_heuristic(routes: Dict[int, List[int]], instance_data: Dict[str, Any], vehicle_capacity: bool = True,
                          verbose: bool = False) -> Dict[str, Any]:
    """
    Runs a production scheduling heuristic for a given set of loader routes and instance data.

    This function generates a detailed schedule for each loader based on their assigned
    routes, beet volumes, productivity rates, and daily loading goals. It simulates
    the loading process over time periods, accounting for travel times and vehicle capacity.

    Args:
        routes (Dict[int, List[int]]): A dictionary mapping loader IDs to their
            specific routes (list of field/location IDs).
        instance_data (Dict[str, Any]): A dictionary containing all necessary instance data,
            including 't_p' (time period length), 'beet_volume', 'loader_data',
            'production_plan', and 'l_tt_matrices' (travel time matrices).
        vehicle_capacity (bool): If True, the heuristic considers vehicle capacity limits.
            Defaults to True.
        verbose (bool): If True, prints detailed information about the heuristic's execution.
            Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the heuristic's results:
            - 'loader_routes' (Dict[int, List[int]]): The input loader routes.
            - 'loader_schedule' (Dict[int, List[List[int]]]): A dictionary where keys are
              loader IDs and values are their generated schedules (list of [from, to] movements
              per time period).
            - 'T' (int): The maximum time horizon observed across all generated loader schedules.
            - 'production_plan' (pd.DataFrame): The production plan used by the heuristic.
    """

    # Data
    print(instance_data.keys())
    t_p = instance_data["t_p"]
    beet_volume = instance_data["beet_volume"]
    loader_data = instance_data["loader_data"]
    L_bar = {row['Maus Nr.']: row["MeanProductivityPerHour"] * t_p
             for index, row in loader_data.iterrows()}

    production_plan = instance_data["production_plan"]

    tau = instance_data["l_tt_matrices"]

    loader_schedule = {}

    # Create Production Schedule
    for loader_id, route in routes.items():
        # select 6 working days
        daily_limit = production_plan[f'{loader_id} Goal'].loc[:5].tolist()

        if verbose:
            print(f"\n=== Start Heuristic Loader {loader_id} ===\n")

            print(f"daily_limit: {daily_limit}")

        loader_schedule[loader_id] = create_production_plan_schedule_tau(route,
                                                                         beet_volume,
                                                                         L_bar[loader_id],
                                                                         daily_limit,
                                                                         t_p, tau,
                                                                         machine_id=loader_id,
                                                                         vehicle_capacity_flag=vehicle_capacity)

    T = max(len(schedule) for schedule in loader_schedule.values())

    print("Finished Heuristic")

    # Create a results dictionary for this scenario
    model_versions_results = {
        "loader_routes": routes,
        "loader_schedule": loader_schedule,
        "T": T,
        "production_plan": production_plan
    }

    return model_versions_results


def run_loader_rolling_flow_TimeAgg(
        current_bracket: Dict[int, List[int]],
        instance_data2h: Dict[str, Any],
        instance_data1h: Dict[str, Any],
        gurobi_model_versions_2h: Dict[str, Any],
        gurobi_model_versions_1h: Dict[str, Any],
        sens: Dict[str, float],
        solver_params: Dict[str, Any],
        base_file_path: str,
        experiment_name: str,
        *,
        heuristic_results2h: Optional[Dict[str, Any]] = None,
        heuristic_results1h: Optional[Dict[str, Any]] = None,
        hotstart_solution: Optional[Dict[str, Any]] = None,
        enforce_solution: Optional[Dict[str, Any]] = None,
        inventory_level: float = 0,
        inventory_flag: bool = True,
        params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Executes a single iteration of the time-aggregated rolling horizon optimization for loaders.

    This function solves both the 2-hour and 1-hour Loader Flow models for a given
    `current_bracket` of fields. It leverages heuristic results for initial routing
    and uses the 2-hour model's solution to prune and warm-start the 1-hour model.

    Args:
        current_bracket (Dict[int, List[int]]): A dictionary mapping loader IDs to
            the list of fields (IDs) assigned to them for the current rolling horizon
            bracket.
        instance_data2h (Dict[str, Any]): The full instance data relevant to the
            2-hour time aggregation model.
        instance_data1h (Dict[str, Any]): The full instance data relevant to the
            1-hour time aggregation model.
        gurobi_model_versions_2h (Dict[str, Any]): Model version parameters for the
            2-hour Gurobi model.
        gurobi_model_versions_1h (Dict[str, Any]): Model version parameters for the
            1-hour Gurobi model.
        sens (Dict[str, float]): Sensitivity scenario parameters (e.g., cost multipliers).
        solver_params (Dict[str, Any]): Gurobi solver parameters.
        base_file_path (str): Base path for saving results.
        experiment_name (str): Name of the current experiment, used for saving files.
        heuristic_results2h (Optional[Dict[str, Any]]): Pre-computed heuristic results
            for the 2-hour model, containing `loader_routes`. If None, the heuristic
            will be re-run. Defaults to None.
        heuristic_results1h (Optional[Dict[str, Any]]): Pre-computed heuristic results
            for the 1-hour model. Currently not directly used for hotstart/enforce
            within this function, but passed for potential future use. Defaults to None.
        hotstart_solution (Optional[Dict[str, Any]]): A pre-computed solution to
            warm-start the 2-hour model. Defaults to None.
        enforce_solution (Optional[Dict[str, Any]]): A pre-computed solution to
            enforce as hard constraints in the 2-hour model. Defaults to None.
        inventory_level (float): The current inventory level of beets at the start
            of this rolling horizon iteration. Defaults to 0.
        inventory_flag (bool): If True, enables inventory-related constraints.
            Defaults to True.
        params (Optional[Dict[str, Any]]): Dictionary of additional or overriding
            parameters for model flags (e.g., `vehicle_capacity_flag`,
            `restricted_flow_flag1h`, etc.). Defaults to None.

    Returns:
        Dict[str, Any]: The detailed results of the 1-hour (fine-grained) Gurobi model
            solution for the current bracket.
    """

    # Set default parameters if not provided or merge with provided ones
    _default_params = {
        'vehicle_capacity_flag': True,
        'restricted_flow_flag1h': True,
        'restricted_flow_flag2h': True,
        'last_period_restricted_flag1h': False,
        'last_period_restricted_flag2h': False,
        'add_min_beet_restriction_flag1h': False,
        'add_min_beet_restriction_flag2h': False,
        'time_restriction': False,
        'verbose': None,
        'v_type': "binary"
    }

    # Merge default params with any provided params, prioritizing provided ones
    resolved_params = {**_default_params, **(params if params is not None else {})}

    # Extract parameters for cleaner access within the function
    vehicle_capacity_flag = resolved_params['vehicle_capacity_flag']
    restricted_flow_flag1h = resolved_params['restricted_flow_flag1h']
    restricted_flow_flag2h = resolved_params['restricted_flow_flag2h']
    last_period_restricted_flag1h = resolved_params['last_period_restricted_flag1h']
    last_period_restricted_flag2h = resolved_params['last_period_restricted_flag2h']
    add_min_beet_restriction_flag1h = resolved_params['add_min_beet_restriction_flag1h']
    add_min_beet_restriction_flag2h = resolved_params['add_min_beet_restriction_flag2h']
    time_restriction = resolved_params['time_restriction']
    verbose = resolved_params['verbose']
    v_type = resolved_params['v_type']

    print("\nStart Inside Gurobi Function Heuristic:\n")

    # Create scenario results to get a T and delay estimation
    print("\n2h Heuristic to get loader routes:\n")
    if not heuristic_results2h:
        heuristic_results2h = run_rolling_heuristic(current_bracket, instance_data2h, verbose=True)

    if verbose:
        print("Finished heuristic")

    # Set model versions
    model_ver_2h = gurobi_model_versions_2h

    # Set T
    T_2h = int(((24 / 2) * 6) + 1)  # 6 Days + 1 set up period

    # Extract heuristic information
    loader_routes_heuristic = heuristic_results2h["loader_routes"]

    instance_raw_2h, derived_2h = prepare_loader_flow_data(instance_data2h,
                                                           model_ver=model_ver_2h,
                                                           sens=sens,
                                                           T_heuristic=T_2h,
                                                           loader_routes_heuristic=loader_routes_heuristic,
                                                           inventory_levels=inventory_level)

    if enforce_solution:
        print("\nEnforce Schedule in run_loader_rolling_flow_TimeAgg:", enforce_solution['loader_schedule'])

    if hotstart_solution:
        print("\nHotstart Schedule in run_loader_rolling_flow_TimeAgg:", hotstart_solution['loader_schedule'])

    print("\n", "=" * 5, "START 2H MODEL", "=" * 5, "\n")
    # Solve Model 2h
    gurobi_2h = solve_loader_flow(
        instance=instance_raw_2h,
        derived=derived_2h,
        name="rolling_timeagg_2h",
        hotstart_solution=hotstart_solution,  # given schedule a nested list
        enforce_solution=enforce_solution,  # given schedule a nested list
        FIXED_SOLVER_PARAMS=solver_params,
        vehicle_capacity_flag=vehicle_capacity_flag,
        restricted_flow_flag=restricted_flow_flag2h,
        last_period_restricted_flag=last_period_restricted_flag2h,
        add_min_beet_restriction_flag=add_min_beet_restriction_flag2h,
        inventory_flag=inventory_flag,
        holidays_flag=True,
        verbose=False,
        base_file_path=base_file_path,
        v_type=v_type)

    # Extract Gurobi's decision variables that store solution
    decision_variables_2h = gurobi_2h["decision_variables"]
    parameters_2h = gurobi_2h["parameters"]

    # Extract last period beets
    last_period_beet_volume = extract_last_period_beet_volume(
        params=parameters_2h,
        decision_variables=decision_variables_2h)

    # Extract Schedule & Get Valid Combinations
    valid_combinations = generate_valid_arcs_for_all_loaders(
        decision_variables=decision_variables_2h,
        ratio=2,  # Quick workaround to be automated to allow for all combinations (4h-2h etc.)
        buffer_pct=derived_2h["buffer_pct"]
    )

    T_1h = int(((24 / 1) * 6) + 1)  # 6 Days + 1 set up period

    # Adapt model versions t_p (workaround may be fixed later)
    model_ver_copy = model_ver_2h.copy()
    model_ver_copy["t_p"] = 1

    # Adapt T calculations
    # T_heuristic_1h = heuristic_results1h[combined_key]["T"]
    # Load 1h Data with valid_comb.
    instance_raw_1h, derived_1h = prepare_loader_flow_data(instance_data1h,
                                                           model_ver=model_ver_copy,
                                                           T_heuristic=T_1h,
                                                           sens=sens,
                                                           loader_routes_heuristic=None,
                                                           valid_arcs_at_time=valid_combinations,
                                                           inventory_levels=inventory_level,
                                                           last_period_beet_volume=last_period_beet_volume)

    print("\n", "=" * 5, "START 1H MODEL", "=" * 5, "\n")

    # Solve Model 1h
    gurobi_1h = solve_loader_flow(
        instance=instance_raw_1h,
        derived=derived_1h,
        name="rolling_timeagg_1h",
        hotstart_solution=None,  # given schedule a nested list
        enforce_solution=None,  # given schedule a nested list
        FIXED_SOLVER_PARAMS=solver_params,
        vehicle_capacity_flag=vehicle_capacity_flag,
        restricted_flow_flag=restricted_flow_flag1h,
        last_period_restricted_flag=last_period_restricted_flag1h,
        add_min_beet_restriction_flag=add_min_beet_restriction_flag1h,
        inventory_flag=inventory_flag,
        holidays_flag=True,
        verbose=False,
        base_file_path=base_file_path,
        v_type=v_type)

    return {
        "opt_type": "time_agg",
        "scenario_results_coarse": gurobi_2h,
        "scenario_results_fine": gurobi_1h
    }


def loader_rolling_flow_experiments_TimeAgg(
        group_id: int,
        scenario_id: str,
        instance_raw_2h: Dict[int, Dict[str, Any]],
        instance_raw_1h: Dict[int, Dict[str, Any]],
        gurobi_model_versions_2h: Dict[str, Any],
        gurobi_model_versions_1h: Dict[str, Any],
        sens: Dict[str, float],
        solver_params: Dict[str, Any],
        region_routes: Dict[int, List[int]],
        base_file_path: str = "../../data/",
        *,
        max_iterations: int = 6,
        params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Manages the overall rolling horizon optimization process for loader scheduling
    using a time-aggregated decomposition approach.

    This function iterates through weekly brackets, updating field volumes and routes,
    and then calls `run_loader_rolling_flow_TimeAgg` for each week to solve the
    optimized schedule. It tracks cumulative results and saves them.

    Args:
        group_id (int): Identifier for the current loader group being processed.
        scenario_id (str): Unique identifier for the current scenario.
        instance_raw_2h (Dict[int, Dict[str, Any]]): Raw instance data structured
            by group_id and scenario_id for the 2-hour models.
        instance_raw_1h (Dict[int, Dict[str, Any]]): Raw instance data structured
            by group_id and scenario_id for the 1-hour models.
        gurobi_model_versions_2h (Dict[str, Any]): Model version parameters for the
            2-hour Gurobi model.
        gurobi_model_versions_1h (Dict[str, Any]): Model version parameters for the
            1-hour Gurobi model.
        sens (Dict[str, float]): Sensitivity scenario parameters.
        solver_params (Dict[str, Any]): Gurobi solver parameters.
        region_routes (Dict[int, List[int]]): Initial long-term routes for each loader
            in the current region/group.
        base_file_path (str): Base path for loading and saving data.
            Defaults to "../../data/".
        max_iterations (int): Maximum number of weekly iterations for the rolling horizon.
            Defaults to 6.
        params (Optional[Dict[str, Any]]): Dictionary of additional or overriding
            parameters for model flags passed to `run_loader_rolling_flow_TimeAgg`.
            Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing weekly results, where keys are
            week identifiers (e.g., "Week_1") and values are dictionaries with
            detailed results, KPIs, and updated states for that week.
    """

    if not params:
        print("\nWARNING: Use standard time agg params!\n")
    #
    # **** 1. Load instance data ****
    #

    instance_data_1h = instance_raw_1h[group_id][scenario_id].copy()
    instance_data_2h = instance_raw_2h[group_id][scenario_id].copy()

    # TODO: this is a quick fix, fix instance data t_p in instance data creation
    instance_data_1h["t_p"] = 1
    instance_data_2h["t_p"] = 2

    #
    # **** 2. Initialise Relevant Parameters ****
    #

    beet_volume = deepcopy(instance_data_1h["beet_volume"])  # Deep copy to prevent modifying original data
    total_volume = sum(beet_volume)

    production_volume_per_day = instance_data_1h["production_volume_per_day"]
    production_plan = deepcopy(instance_data_1h["production_plan"])
    L = instance_data_1h["L"]

    # Initialize variables
    updated_volumes = beet_volume
    updated_routes = deepcopy(region_routes)
    updated_production_plan = deepcopy(production_plan)
    inventory_level = 0

    #
    # **** start rolling while loop ****
    #

    weekly_results = {}  # To store results for each week with week identifiers as keys
    iteration = 0

    # Extract loader ids from dict
    loader_ids = list(region_routes.keys())

    while total_volume > 0 and iteration < max_iterations and all(
            len(region_routes[loader_id]) > 0 for loader_id in loader_ids) and len(L) >= 2:

        # Check conditions to log reason for stopping
        if total_volume <= 0:
            print("Stopping: Total volume has been reduced to zero or below.")
        elif iteration >= max_iterations:
            print("Stopping: Maximum number of iterations reached.")
        elif not all(len(region_routes[loader_id]) > 0 for loader_id in loader_ids):
            print("Stopping: One or more loaders have no remaining routes.")
        elif not len(L) >= 2:
            print("Stopping: Less than two loaders left")

        # Start loop logic
        week_identifier = f"Week_{iteration + 1}"
        print(f"\n=== {week_identifier} ===")

        # **** 3. Split Sequence into Brackets ****
        weekly_brackets = create_weekly_brackets(updated_routes, updated_volumes, updated_production_plan,
                                                 verbose=False)

        current_bracket = {}
        print("Current Weekly Brackets:")
        for loader, brackets in weekly_brackets.items():
            if brackets:
                current_bracket[loader] = brackets[0][1]
                print(f"Loader: {loader} - Bracket: {brackets[0][1]}")
            else:
                print(f"Loader: {loader} - No brackets available.")

        if not current_bracket:
            print("No brackets to process. Exiting loop.")
            break  # No brackets to process

        print("\nUpdated Production Plan Head 7")
        print(updated_production_plan.head(7))

        # Update field sets
        I = set()
        Il = {}

        # Iterate over the dictionary values
        for loader_id, fields in current_bracket.items():
            Il[loader_id] = set(fields)  # Add all values to set I
            I.update([num for num in fields if num != 0])
            print(f"Loader {loader_id} number of fields: {len(Il[loader_id])}")

        # Update data
        instance_data_1h["I"] = I
        instance_data_2h["I"] = I
        instance_data_1h["Il"] = Il
        instance_data_2h["Il"] = Il

        # Custom EXPERIMENT

        # Custom Holidays:

        # instance_data_1h["holidays"] = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        # 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
        # instance_data_2h["holidays"] = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        # Adapt production plan
        # production_plan = instance_data_2h["production_plan"].copy()

        # production_plan.loc[1, 'Holiday'] = True
        # production_plan.loc[1, production_plan.columns.difference(['Date', 'Holiday'])] = 0

        # instance_data_1h["production_plan"] = production_plan.copy()
        # instance_data_2h["production_plan"] = production_plan.copy()

        # inventory_level = 50 #4500

        # **** 4. Optimize ****

        # Heuristic

        # Run Heuristics 2h
        heuristic_results_2h = run_rolling_heuristic(current_bracket, instance_data_2h, verbose=True)

        # Run Heuristic 1h
        heuristic_results_1h = run_rolling_heuristic(current_bracket, instance_data_1h, verbose=True)

        # Enforce 1h Heuristic

        T_1h = int(((24 / 1) * 6) + 1)  # 6 Days + 1 set up period

        # Adapt model versions t_p (workaround may be fixed later)
        model_ver_copy = gurobi_model_versions_1h.copy()
        model_ver_copy["t_p"] = 1
        loader_routes_heuristic = heuristic_results_1h["loader_routes"]
        # loader_schedule = heuristic_results_1h["loader_schedule"]

        # Adapt T calculations
        # T_heuristic_1h = heuristic_results1h[combined_key]["T"]
        # Load 1h Data with valid_comb.
        instance_raw_1h, derived_1h = prepare_loader_flow_data(instance_data_1h,
                                                               sens=sens,
                                                               model_ver=model_ver_copy,
                                                               T_heuristic=T_1h,
                                                               loader_routes_heuristic=loader_routes_heuristic,
                                                               inventory_levels=inventory_level)

        # Enforce Heuristic

        print("\n Start Benchmark: Heuristic Solution!! \n")

        print("1h heuristic schedule:", heuristic_results_1h["loader_schedule"])

        gurobi_heuristic_results_1h = solve_loader_flow(
            instance=instance_raw_1h,
            derived=derived_1h,
            name="test_TimeAgg",
            hotstart_solution=None,  # given schedule a nested list
            enforce_solution=heuristic_results_1h,  # heuristic_results_1h,  # given schedule a nested list
            FIXED_SOLVER_PARAMS=solver_params,
            vehicle_capacity_flag=True,
            restricted_flow_flag=False,
            last_period_restricted_flag=True,
            add_min_beet_restriction_flag=False,
            inventory_flag=True,
            holidays_flag=True,
            verbose=False,
            base_file_path=base_file_path,
            v_type="binary")

        print("\n", "*" * 5, "START TIMEAGG", "*" * 5, "\n")

        # Run Time Agg using 2h hotstart
        gurobi_results_TimeAgg = run_loader_rolling_flow_TimeAgg(
            current_bracket,
            instance_data2h=instance_data_2h,  # Messy workaround, to be fixed (think about summarise in one dict)
            instance_data1h=instance_data_1h,
            gurobi_model_versions_2h=gurobi_model_versions_2h,
            gurobi_model_versions_1h=gurobi_model_versions_1h,
            sens=sens,
            solver_params=solver_params,
            base_file_path=base_file_path,
            experiment_name="rolling_time_agg",
            heuristic_results2h=heuristic_results_2h,
            heuristic_results1h=heuristic_results_1h,
            hotstart_solution=heuristic_results_2h,
            inventory_level=inventory_level,
            enforce_solution=None,
            params=params)

        # KPI Comparison
        kpi_comparison = create_kpi_comparison(
            gurobi_results_TimeAgg,  # Pass the whole object
            gurobi_heuristic_results_1h,
            flow_type="load_flow"
        )

        print("KPI Comparison:")
        print(kpi_comparison)

        # **** 5. Save Results and Update Long Sequence ****

        # Update volumes, routes, and production plan based on Gurobi results
        fine_solution_vars = gurobi_results_TimeAgg["scenario_results_fine"].get("decision_variables", {})
        updated_volumes, updated_routes, updated_production_plan, updated_inventory = update_volumes_and_routes(
            fine_solution_vars,
            updated_volumes,
            updated_routes,
            updated_production_plan,
            load_inventory=True
        )

        print("Inventory Level: ", inventory_level)
        print("Production Volume Per Day: ", production_volume_per_day)

        inventory_level = updated_inventory - production_volume_per_day
        inventory_level = 0 if inventory_level < 1 else inventory_level

        print("Updated Inventory Level: ", updated_inventory)

        # Update beet volumes and prodution plan in instant data, to align gurobi variables
        instance_data_1h["beet_volume"] = deepcopy(updated_volumes)
        instance_data_2h["beet_volume"] = deepcopy(updated_volumes)
        instance_data_1h["production_plan"] = deepcopy(updated_production_plan)
        instance_data_2h["production_plan"] = deepcopy(updated_production_plan)

        instance_data_1h["holidays"] = production_plan_holiday_to_t(
            updated_production_plan,
            periods_per_day=int(24 / 1)  # TODO: Fix if static t_p dynamic systems are needed
        )

        instance_data_2h["holidays"] = production_plan_holiday_to_t(
            updated_production_plan,
            periods_per_day=int(24 / 2)  # TODO: Fix if static t_p dynamic systems are needed
        )

        print("Updated Region Routes:")
        print(updated_routes)

        # Collect weekly results
        week_result = {
            "iteration": iteration + 1,
            "heuristic_results": deepcopy(heuristic_results_1h),
            "gurobi_results": deepcopy(gurobi_results_TimeAgg),
            "gurobi_heuristic_results_pl": deepcopy(gurobi_heuristic_results_1h),
            "kpi_comparison": kpi_comparison.copy(),
            "updated_volumes": deepcopy(updated_volumes),
            "updated_routes": deepcopy(updated_routes),
            "updated_production_plan": deepcopy(updated_production_plan),
            "weekly_result": deepcopy(inventory_level)
        }
        weekly_results[week_identifier] = week_result

        # Update total_volume
        total_volume = sum(updated_volumes)
        print(f"Total Remaining Volume: {total_volume}")

        # Increment iteration
        iteration += 1

    if iteration >= max_iterations:
        print(f"Reached maximum iterations ({max_iterations}). Some fields may not be processed completely.")
    else:
        print("All fields have been processed successfully.")

    # **** 6. Save Weekly Results as Pickle ****
    today_date = datetime.now().strftime('%Y%m%d')

    pickle_file_path = os.path.join(base_file_path, f'results/reporting/weekly_results_{today_date}.pkl')
    try:
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(weekly_results, pickle_file)
        print(f"Weekly results successfully saved to {pickle_file_path}")
    except Exception as e:
        print(f"Error saving weekly results to pickle file: {e}")

    # Optional: Save KPI Comparisons to Excel (Uncomment if needed)
    """
    excel_file_path = os.path.join(results_dir, 'weekly_kpi_comparison.xlsx')
    try:
        with pd.ExcelWriter(excel_file_path) as writer:
            for week_id, week_data in weekly_results.items():
                kpi = week_data['kpi_comparison']
                kpi.to_excel(writer, sheet_name=week_id)
        print(f"KPI comparisons successfully saved to {excel_file_path}")
    except Exception as e:
        print(f"Error saving KPI comparisons to Excel file: {e}")
    """

    return weekly_results
