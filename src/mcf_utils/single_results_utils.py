import pandas as pd
import numpy as np
import pickle
import os


def extract_machinery_info(decision_variables, machinery_type):
    I = {i for i, _, _, _ in decision_variables['beet_flow_b'].keys()}
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    df_machinery = pd.DataFrame(index=list(I), columns=list(T), data='')
    machinery_info = {}
    for key, value in decision_variables.items():
        if key.startswith(machinery_type):  # Filter by machinery type ('harvester_flow_x' or 'loader_flow_x')
            for (h, i, j, t), x_val in value.items():
                if i == j and x_val > 0.5:  # Focus on (i, i) and check if machinery is scheduled
                    if (i, t) not in machinery_info:
                        machinery_info[(i, t)] = []
                    machinery_info[(i, t)].append(f"{machinery_type[0].upper()}{h}")  # 'H' or 'L' + ID
    return machinery_info, df_machinery


def extract_machinery_schedule_idle_twolevel(decision_variables):
    # Check for idle/working presence
    idle_present = 'idle' in decision_variables and 'working' in decision_variables

    # Collect all time periods from loader_flow_x (or from relevant var)
    # This ensures we don't miss times that appear only in loader_flow_x
    all_vars = ['beet_flow_b', 'loader_flow_x']  # or others if needed
    T = sorted({
        t for var_name in all_vars if var_name in decision_variables
        for (_, _, _, t) in decision_variables[var_name].keys()
    })

    # Initialize structure: machinery_schedule[machinery_id][t] -> one item: "idle", (i,i), or (i,j)
    machinery_schedule = {}

    # 1) First incorporate idle/working if present
    if idle_present:
        # We'll loop over "working" or "idle" to detect times
        for (machine_id, i, t), idle_val in decision_variables['idle'].items():
            if idle_val > 0.5:
                # Mark "idle"
                mach_id_str = f"{machine_id}"  # you might build something like "L{machine_id}"
                if mach_id_str not in machinery_schedule:
                    machinery_schedule[mach_id_str] = {time: None for time in T}
                machinery_schedule[mach_id_str][t] = "idle"

        for (machine_id, i, t), work_val in decision_variables['working'].items():
            if work_val > 0.5:
                # Mark (i,i)
                mach_id_str = f"{machine_id}"
                if mach_id_str not in machinery_schedule:
                    machinery_schedule[mach_id_str] = {time: None for time in T}
                machinery_schedule[mach_id_str][t] = (i, i)

    # 2) Next incorporate loader_flow_x for movement
    if 'loader_flow_x' in decision_variables:
        for (machine_id, i, j, t), x_val in decision_variables['loader_flow_x'].items():
            if x_val > 0.5:
                # If we haven't tracked the machine yet, init
                mach_id_str = f"{machine_id}"
                if mach_id_str not in machinery_schedule:
                    machinery_schedule[mach_id_str] = {time: None for time in T}

                # If there's nothing assigned yet at time t, store (i, j).
                # If it's already "idle" or (i,i), you may decide how to handle that conflict.
                if machinery_schedule[mach_id_str][t] is None:
                    machinery_schedule[mach_id_str][t] = (i, j)

    # Build a MultiIndex for columns
    col_index = pd.MultiIndex.from_tuples(
        [(t, pos_type) for t in T for pos_type in ["start_pos", "end_pos"]],
        names=["time", "status"]
    )

    # Create the DataFrame
    df_machinery_schedule = pd.DataFrame(
        index=sorted(machinery_schedule.keys()),
        columns=col_index,
        dtype=object
    )

    # Fill the DataFrame
    for m_id, t_dict in machinery_schedule.items():
        for t in T:
            entry = t_dict[t]
            if entry is None:
                df_machinery_schedule.loc[m_id, (t, "start_pos")] = None
                df_machinery_schedule.loc[m_id, (t, "end_pos")] = None
            elif entry == "idle":
                df_machinery_schedule.loc[m_id, (t, "start_pos")] = "idle"
                df_machinery_schedule.loc[m_id, (t, "end_pos")] = "idle"
            else:
                # entry is a tuple (i, j)
                i_val, j_val = entry
                df_machinery_schedule.loc[m_id, (t, "start_pos")] = i_val
                df_machinery_schedule.loc[m_id, (t, "end_pos")] = j_val

    return df_machinery_schedule


def extract_machinery_schedule_idle(decision_variables, machinery_types):
    # Check if idle time is part of the decision variables
    idle_present = 'idle' in decision_variables and 'working' in decision_variables

    # Extract unique time slots
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    # Initialize DataFrame for machinery schedule
    df_machinery_schedule = pd.DataFrame(columns=list(T))

    # Initialize a dictionary to hold all machinery schedules
    machinery_schedule = {}

    # Loop over each type of machinery (harvester, loader, etc.)
    for machinery_type in machinery_types:
        # Loop through each entry in decision_variables related to the current machinery type
        for key, value in decision_variables.items():
            if key.startswith(machinery_type):
                for (machine_id, i, j, t), x_val in value.items():
                    if i == j and x_val > 0.5:  # Machinery is working in the same position (i, i)
                        machinery_id = f"{machinery_type[0].upper()}{machine_id}"

                        # Ensure that each machinery_id and time are initialized in machinery_schedule
                        if machinery_id not in machinery_schedule:
                            machinery_schedule[machinery_id] = {time: [] for time in T}

                        # Check if idle and working times are available
                        if idle_present:
                            idle_val = decision_variables['idle'].get((machine_id, i, t), 0)
                            working_val = decision_variables['working'].get((machine_id, i, t), 0)

                            # Ensure only one of them is 1, if idle_val == 1, then add "idle", else add the position i
                            if idle_val > 0.5:
                                machinery_schedule[machinery_id][t].append("idle")
                            elif working_val > 0.5:
                                machinery_schedule[machinery_id][t].append(i)
                        else:
                            # If no idle/working time, default to the previous behavior
                            machinery_schedule[machinery_id][t].append(i)

    # Convert the dictionary schedule into the DataFrame, adding to the existing DataFrame
    for machinery_id, times in machinery_schedule.items():
        df_machinery_schedule.loc[machinery_id] = pd.Series(
            {t: ', '.join(map(str, set(states))) for t, states in times.items()})

    return df_machinery_schedule


def extract_machinery_schedule(decision_variables, machinery_types):
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    # Initialize DataFrame for machinery schedule
    df_machinery_schedule = pd.DataFrame(columns=list(T))

    # Initialize a dictionary to hold all machinery schedules
    machinery_schedule = {}

    # Loop over each type of machinery (harvester, loader, etc.)
    for machinery_type in machinery_types:
        # Loop through each entry in decision_variables related to the current machinery type
        for key, value in decision_variables.items():
            if key.startswith(machinery_type):
                for (machine_id, i, j, t), x_val in value.items():
                    if i == j and x_val > 0.5:  # Focus on (i, i) and check if machinery is scheduled
                        machinery_id = f"{machinery_type[0].upper()}{machine_id}"

                        # Ensure that each machinery_id and time are initialized in machinery_schedule
                        if machinery_id not in machinery_schedule:
                            machinery_schedule[machinery_id] = {time: [] for time in T}

                        # Record the position at the current time t
                        machinery_schedule[machinery_id][t].append(i)

    # Convert the dictionary schedule into the DataFrame, adding to existing DataFrame
    for machinery_id, times in machinery_schedule.items():
        df_machinery_schedule.loc[machinery_id] = pd.Series(
            {t: ', '.join(map(str, set(states))) for t, states in times.items()})

    return df_machinery_schedule


def beet_movement_info(decision_variables, flow_type):
    # Extract I and T from decision variables
    I = {i for i, _, _, _ in decision_variables['beet_flow_b'].keys()}
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    if flow_type == "full_flow":
        # Initialize totals dictionary with initial values set to 0
        totals = {
            'Volume_Field': {t: 0 for t in T},
            'Harvested_t': {t: 0 for t in T},
            'Total_Harvested': {t: 0 for t in T},
            'Harvested_state': {t: 0 for t in T},
            'Loaded_t': {t: 0 for t in T},
            'Total_Loaded': {t: 0 for t in T},
            'Inventory_t': {t: 0 for t in T},
            'Processed_t': {t: 0 for t in T},
            'Total_Processed': {t: 0 for t in T}
        }

        # Initialize empty variables for accumulating totals
        total_harvested = 0
        total_loaded = 0
        total_processed = 0

        # Populate totals based on beet_flow_b values
        for t in T:
            totals['Volume_Field'][t] = sum(decision_variables["beet_flow_b"][(i, 0, 0, t)] for i in I)
            totals['Harvested_t'][t] = sum(decision_variables["beet_flow_b"][(i, 0, 1, t)] for i in I)
            totals['Harvested_state'][t] = sum(decision_variables["beet_flow_b"][(i, 1, 1, t)] for i in I)

            # Checking if any loads are available to be summed
            if any((i, 1, 2, t) in decision_variables['beet_flow_b'] for i in I):
                totals['Loaded_t'][t] = sum(decision_variables["beet_flow_b"][(i, 1, 2, t)] for i in I)
            else:
                totals['Loaded_t'][t] = 0

            if any((i, 1, 3, t) in decision_variables['beet_flow_b'] for i in I):
                totals['Loaded_t'][t] += sum(decision_variables["beet_flow_b"][(i, 1, 3, t)] for i in I)
                totals['Inventory_t'][t] = sum(decision_variables["beet_flow_b"][(i, 2, 2, t)] for i in I)
                totals['Processed_t'][t] = sum(decision_variables["beet_flow_b"][(i, 1, 3, t)] +
                                               decision_variables["beet_flow_b"][(i, 2, 3, t)] for i in I)
            else:
                totals['Inventory_t'][t] = 0
                totals['Processed_t'][t] = 0

            # Accumulate totals
            total_harvested += totals['Harvested_t'][t]
            total_loaded += totals['Loaded_t'][t]
            total_processed += totals['Processed_t'][t]

            # Update the totals in the dictionary
            totals['Total_Harvested'][t] = total_harvested
            totals['Total_Loaded'][t] = total_loaded
            totals['Total_Processed'][t] = total_processed

    elif flow_type == "load_flow":
        # Initialize totals dictionary with initial values set to 0 for load_flow
        totals = {
            'Volume_Field': {t: 0 for t in T},
            'Loaded_t': {t: 0 for t in T},
            'Total_Loaded': {t: 0 for t in T},
            'Inventory_t': {t: 0 for t in T},
            'Processed_t': {t: 0 for t in T},
            'Total_Processed': {t: 0 for t in T}
        }

        # Initialize empty variables for accumulating totals
        total_loaded = 0
        total_processed = 0

        # Populate totals based on beet_flow_b values for load_flow
        for t in T:
            totals['Volume_Field'][t] = sum(decision_variables["beet_flow_b"][(i, 0, 0, t)] for i in I)

            # Direct loading to inventory or production
            if any((i, 0, 2, t) or (i, 0, 1, t) in decision_variables['beet_flow_b'] for i in I):
                totals['Loaded_t'][t] = sum(decision_variables["beet_flow_b"][(i, 0, 2, t)] for i in I) + \
                                        sum(decision_variables["beet_flow_b"][(i, 0, 1, t)] for i in I)
                totals['Inventory_t'][t] = sum(decision_variables["beet_flow_b"][(i, 1, 1, t)] for i in I)
                totals['Processed_t'][t] = sum(decision_variables["beet_flow_b"][(i, 0, 2, t)] +
                                               decision_variables["beet_flow_b"][(i, 1, 2, t)] for i in I)
            else:
                totals['Loaded_t'][t] = 0
                totals['Inventory_t'][t] = 0
                totals['Processed_t'][t] = 0

            # Accumulate totals
            total_loaded += totals['Loaded_t'][t]
            total_processed += totals['Processed_t'][t]

            # Update the totals in the dictionary
            totals['Total_Loaded'][t] = total_loaded
            totals['Total_Processed'][t] = total_processed

    # Convert to DataFrame for easier visualization and manipulation
    df_totals = pd.DataFrame(totals)

    # Display the transposed DataFrame for a better overview of each time period
    df_rounded = np.round(df_totals.T, 0)

    return df_rounded


def extract_field_yield(decision_variables):
    # Extract I and T from decision variables
    I = {i for i, _, _, _ in decision_variables['beet_flow_b'].keys()}
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    # Initialize DataFrame with fields as index and time periods as columns, initializing with zero
    df_field_yield = pd.DataFrame(index=list(I), columns=list(T), data=0)

    # Populate df_field_yield based on decision variables, filling each field and time period
    for t in T:
        for i in I:
            df_field_yield.at[i, t] = np.round(decision_variables["beet_flow_b"][(i, 0, 0, t)], 0)

    return df_field_yield


def extract_machine_costs(decision_variables, machinery_type, cost_matrix, machinery_set):
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    # Determine the prefix based on machinery type
    prefix = 'H' if 'harvester' in machinery_type else 'L'

    # Construct the MultiIndex with customized labels based on machinery type
    cost_types = ['travel', 'operation']
    machine_labels = [f"{prefix}{i}" for i in machinery_set]
    index = pd.MultiIndex.from_product([machine_labels, cost_types], names=['Machine', 'Cost Type'])
    df_costs = pd.DataFrame(index=index, columns=list(T), data=0.0).fillna(0.0)

    # Populate the DataFrame
    for (machine_id, i, j, t), item in decision_variables[machinery_type].items():
        if item > 0.5:  # Action worth costing
            cost = cost_matrix[machine_id, i, j, t]
            cost_type = 'travel' if i != j else 'operation'
            machine_label = f"{prefix}{machine_id}"  # Create the label for the index

            # Set the cost in the DataFrame under the correct machine label and cost type
            df_costs.at[(machine_label, cost_type), t] = cost

    return df_costs


def extract_machine_costs_idle(decision_variables, machinery_type, cost_matrix, machinery_set):
    # Check if idle and working time are part of the decision variables
    idle_present = 'idle' in decision_variables and 'working' in decision_variables

    # Extract unique time slots
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    # Determine the prefix based on machinery type
    prefix = 'H' if 'harvester' in machinery_type else 'L'

    # Construct the MultiIndex with customized labels based on machinery type
    cost_types = ['travel', 'operation']
    machine_labels = [f"{prefix}{i}" for i in machinery_set]
    index = pd.MultiIndex.from_product([machine_labels, cost_types], names=['Machine', 'Cost Type'])
    df_costs = pd.DataFrame(index=index, columns=list(T), data=0.0).fillna(0.0)

    # Populate the DataFrame
    for (machine_id, i, j, t), item in decision_variables[machinery_type].items():
        if item > 0.5:  # Action worth costing
            cost = cost_matrix[machine_id].at[i, j]
            cost_type = 'travel' if i != j else 'operation'
            machine_label = f"{prefix}{machine_id}"  # Create the label for the index

            # Check for idle/working status if idle/working is present
            if cost_type == 'operation' and idle_present:
                idle_val = decision_variables['idle'].get((machine_id, i, t), 0)
                working_val = decision_variables['working'].get((machine_id, i, t), 0)

                # Only add operation cost if the machine is working, not idle
                if working_val > 0.5:
                    df_costs.at[(machine_label, cost_type), t] = cost
            else:
                # For travel or when idle/working is not present, assign the cost directly
                df_costs.at[(machine_label, cost_type), t] = cost

    return df_costs


def extract_machine_costs_with_partial(decision_variables, machinery_type, cost_matrix, machinery_set, op_cost=230):
    # Check if idle and working status are available
    idle_present = 'idle' in decision_variables and 'working' in decision_variables

    # Get unique time periods from beet_flow_b keys (assuming all models use the same T)
    T = {t for (_, _, _, t) in decision_variables['beet_flow_b'].keys()}

    # Determine machine label prefix
    prefix = 'H' if 'harvester' in machinery_type.lower() else 'L'

    # Define cost types: travel cost, full operation cost, and partial work cost.
    cost_types = ['travel', 'operation', 'partial_work']
    machine_labels = [f"{prefix}{i}" for i in machinery_set]
    index = pd.MultiIndex.from_product([machine_labels, cost_types], names=['Machine', 'Cost Type'])
    df_costs = pd.DataFrame(index=index, columns=list(T), data=0.0).fillna(0.0)

    # Process binary decisions (x): assign travel and operation cost
    for (machine_id, i, j, t), item in decision_variables[machinery_type].items():
        if item > 0.5:  # decision is “active”
            cost = cost_matrix[machine_id].at[i, j]
            machine_label = f"{prefix}{machine_id}"
            if i != j:
                # travel arc cost (we assume travel cost is provided in cost_matrix)
                df_costs.at[(machine_label, 'travel'), t] = cost
            else:
                # for staying at the same location, check working/idle status
                if idle_present:
                    working_val = decision_variables['working'].get((machine_id, i, t), 0)
                    if working_val > 0.5:
                        df_costs.at[(machine_label, 'operation'), t] = cost
                else:
                    df_costs.at[(machine_label, 'operation'), t] = cost

    # Now process the partial work variables (only for travel arcs, i != j)
    for (machine_id, j, i, t), value in decision_variables['y_in'].items():
        if i != j and value > 1e-5:  # avoid numerical noise
            machine_label = f"{prefix}{machine_id}"
            df_costs.at[(machine_label, 'partial_work'), t] += op_cost * value

    for (machine_id, i, j, t), value in decision_variables['y_out'].items():
        if i != j and value > 1e-5:
            machine_label = f"{prefix}{machine_id}"
            df_costs.at[(machine_label, 'partial_work'), t] += op_cost * value

    # Adjust travel cost by subtracting partial work cost
    for machine in machine_labels:
        for t in T:
            df_costs.at[(machine, 'travel'), t] -= df_costs.at[(machine, 'partial_work'), t]

    return df_costs


def create_machine_cost_parameter_df(cost_matrix):
    """
    Create a cost parameter DataFrame from a dictionary of custom cost matrices, each stored as a DataFrame.

    Parameters:
    cost_matrix (dict): A dictionary where the keys are machine identifiers and the values are DataFrames
                        representing the cost matrices for each machine. The DataFrame has indices and columns
                        corresponding to 'From Location' and 'To Location'.

    Returns:
    pd.DataFrame: A DataFrame with a MultiIndex (Machine, From Location) and columns for 'To Location'.
    """

    # Create lists for MultiIndex and columns
    machine_index = []
    location_index = pd.Index([])  # Initialize empty Index for To Locations

    # Initialize empty DataFrame to store costs
    df_costs = pd.DataFrame()

    # Iterate through the dictionary to extract cost matrices for each machine
    for machine in cost_matrix:
        machine_df = cost_matrix[machine]
        from_locations = machine_df.index
        to_locations = machine_df.columns

        # Append machine and from_location combinations to create a multi-index structure
        machine_index.extend([(machine, from_loc) for from_loc in from_locations])

        # Set the location index only once (assuming it's consistent across machines)
        if location_index.empty:
            location_index = to_locations

        # Append current machine's cost data to the overall DataFrame
        df_costs = pd.concat([df_costs, machine_df])

    # Create MultiIndex for the DataFrame's index
    multi_index = pd.MultiIndex.from_tuples(machine_index, names=['Machine', 'From Location'])

    # Set the DataFrame's index to the MultiIndex and columns to location_index (To Location)
    df_costs.index = multi_index
    df_costs.columns = location_index

    return df_costs


def calculate_revenue_and_unmet_demand(decision_variables, sugar_price, sc, flow_type):
    I = {i for i, _, _, _ in decision_variables['beet_flow_b'].keys()}
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}

    if flow_type == "full_flow":
        # Initialize totals dictionary with initial values set to 0
        totals = {'Harvested_t': {t: 0 for t in T},
                  'Sugar_t': {t: 0 for t in T},
                  'Revenue_t': {t: 0 for t in T},
                  'UD_t': {t: 0 for t in T}
                  }
        # Populate totals based on beet_flow_b values
        for t in T:
            totals['Harvested_t'][t] = sum(decision_variables["beet_flow_b"][(i, 0, 1, t)] for i in I)
            # Loop to match field and sugar concentration
            daily_sugar_yield = sum(decision_variables["beet_flow_b"][(i, 0, 1, t)] * sc[i - 1, t] for i in I)

            totals['Sugar_t'][t] = daily_sugar_yield
            totals['Revenue_t'][t] = daily_sugar_yield * sugar_price
            totals['UD_t'][t] = decision_variables['unmet_demand'].get(t, 0)

    if flow_type == "load_flow":
        # Initialize totals dictionary with initial values set to 0
        totals = {'Loaded_t': {t: 0 for t in T},
                  'Sugar_t': {t: 0 for t in T},
                  'Revenue_t': {t: 0 for t in T},
                  'UD_t': {t: 0 for t in T}
                  }
        # Populate totals based on beet_flow_b values
        for t in T:
            totals['Loaded_t'][t] = sum(
                decision_variables["beet_flow_b"][(i, 0, 1, t)] + decision_variables["beet_flow_b"][(i, 0, 2, t)] for i
                in I)
            # Loop to match field and sugar concentration
            if isinstance(sc, dict):
                daily_sugar_yield = sum(
                    (decision_variables["beet_flow_b"][(i, 0, 1, t)] + decision_variables["beet_flow_b"][
                        (i, 0, 2, t)]) *
                    sc[i] for i in I)
            elif isinstance(sc, float):
                daily_sugar_yield = sum(
                    (decision_variables["beet_flow_b"][(i, 0, 1, t)] + decision_variables["beet_flow_b"][
                        (i, 0, 2, t)]) * sc for i in I)
            else:
                raise ValueError("SC is neither dict nor float type")

            # print(daily_sugar_yield, sugar_price)  # Debug print, can be removed in production

            totals['Sugar_t'][t] = daily_sugar_yield
            totals['Revenue_t'][t] = daily_sugar_yield * sugar_price
            totals['UD_t'][t] = decision_variables['unmet_demand'].get(t, 0)

    # Convert to DataFrame for easier visualization and manipulation
    df_totals = pd.DataFrame(totals)

    # Display the transposed DataFrame for a better overview of each time period
    df_rounded = np.round(df_totals.T, 2)

    return df_rounded


def load_results(file_path):
    with open(file_path, 'rb') as result_file:
        results = pickle.load(result_file)
    return results


def create_all_scheduling_results(decision_variables, c_l, L, sugar_price, sugar_concentration, flow_type,
                                  c_h=None, H=None, tau=None, L_bar=None):
    results = {}

    # Machine Schedule
    machinery_types = ['harvester_flow_x', 'loader_flow_x']
    results['df_schedule'] = extract_machinery_schedule(decision_variables, machinery_types)

    # Get an arbitrary element from the set L
    any_loader = next(iter(L))

    # Machinery Operations Costs
    results['df_loader_cost'] = extract_machine_costs_with_partial(
        decision_variables=decision_variables,
        machinery_type='loader_flow_x',
        cost_matrix=c_l,
        machinery_set=L,
        op_cost=c_l[any_loader].iloc[1, 1]
    )

    if c_h is not None:
        results['df_harvest_cost'] = extract_machine_costs_with_partial(
            decision_variables=decision_variables,
            machinery_type='harvester_flow_h',
            cost_matrix=c_h,
            machinery_set=H,
            op_cost=c_h.iloc[1, 1]
        )
        # Join based on columns
        results['df_machinery_cost'] = pd.concat([results['df_harvest_cost'], results['df_loader_cost']], axis=0)

    else:
        results['df_machinery_cost'] = results['df_loader_cost']

    # Machinery Cost Matrix
    if c_h is not None:
        results['df_harvester_cost_matrix'] = create_machine_cost_parameter_df(c_h)
    # results['df_loader_cost_matrix'] = create_machine_cost_parameter_df(c_l)

    # Beet Movement
    results['df_beet_movement'] = beet_movement_info(decision_variables, flow_type=flow_type)

    # Beet yields per field
    results['df_field_yield'] = extract_field_yield(decision_variables)

    # Revenue Unmet Demand
    results['revenue_unmet_df'] = calculate_revenue_and_unmet_demand(decision_variables, sugar_price,
                                                                     sugar_concentration, flow_type=flow_type)
    #  Include hidden idle insights if tau and productivity are
    if tau is not None and L_bar is not None:
        results['df_hidden_idle_insights'] = create_hidden_idle_insights_table(decision_variables, tau, L_bar)

    return results


def create_hidden_idle_insights_table(decision_variables, tau, L_bar):
    # Get T
    T = {t for _, _, _, t in decision_variables['beet_flow_b'].keys()}
    # Identify machine IDs from 'working' and 'loader_flow_x'
    machine_ids = set()
    for (l, i, t), val in decision_variables.get('working', {}).items():
        machine_ids.add(l)
    for (l, i, j, t), val in decision_variables.get('loader_flow_x', {}).items():
        machine_ids.add(l)

    # Initialize per-machine metrics.
    metrics = {l: {
        'Time_Periods': len(T),
        'Depot_Periods': 0,
        'Onsite_Work_Periods': 0,  # Count of periods with working==1 when machine is on-site (i == j, i != 0)
        'Onsite_Idle_Periods': 0,  # Count of periods with idle==1 when on-site (i == j, i != 0)
        'Travel_Periods': 0,  # Count of travel arcs (loader_flow_x with i != j), including depot moves.
        'Partial_Work_Travel': 0.0,  # Sum of partial work (y_in + y_out) on travel arcs.
        'Travel_Idle_Time': 0.0,  # Sum of unused overhang time on travel arcs.
        'Moved_Beets': 0,
        'Actual_Travel': 0,
        'Actual_Work': 0
    } for l in machine_ids}

    # Depot Periods:
    for (l, i, j, t), val in decision_variables.get('loader_flow_x', {}).items():
        if val > 0.5 and i == 0 and j == 0:
            metrics[l]['Depot_Periods'] += 1

    # Work Periods:

    # Process on-site work and idle (only count if not at depot, i != 0)
    for (l, i, t), val in decision_variables.get('working', {}).items():
        if val > 0.5 and i != 0:
            metrics[l]['Onsite_Work_Periods'] += 1

    for (l, i, t), val in decision_variables.get('idle', {}).items():
        if val > 0.5 and i != 0:
            metrics[l]['Onsite_Idle_Periods'] += 1

    # Actual Travel & Partial Work:

    # For arcs with i != j, if neither endpoint is depot (i, j != 0), count as travel period.
    for (l, i, j, t), val in decision_variables.get('loader_flow_x', {}).items():
        if val > 0.5 and i != j:
            # Exclude arcs that start or end at depot (node 0)

            metrics[l]['Travel_Periods'] += 1

            # Compute partial work on this travel arc.
            y_in_val = decision_variables.get('y_in', {}).get((l, i, j, t), 0) + decision_variables.get('y_in', {}).get(
                (l, j, i, t), 0)
            y_out_val = decision_variables.get('y_out', {}).get((l, i, j, t), 0) + decision_variables.get('y_out',
                                                                                                          {}).get(
                (l, j, i, t), 0)
            partial = y_in_val + y_out_val
            metrics[l]['Partial_Work_Travel'] += partial

            # Available overhang time is the time left in the period after travel.
            # Retrieve travel time from tau, default to 0 if unavailable.
            try:
                travel_time = tau[l].at[i, j]
            except Exception:
                travel_time = 0.0
            available_overhang = max(0, 1 - travel_time)
            metrics[l]['Actual_Travel'] += travel_time
            # Travel idle time is the unused portion of available overhang.
            travel_idle = max(0, available_overhang - partial)
            metrics[l]['Travel_Idle_Time'] += np.round(travel_idle, 5)

    # Actual work estimation using beet flow

    for (l, i, j, t), val in decision_variables.get('loader_flow_x', {}).items():
        if val > 0.5 and t > 0:
            if i != j:
                b_prev = decision_variables["beet_flow_b"].get((i, 0, 0, t - 1), 0) + decision_variables[
                    "beet_flow_b"].get((j, 0, 0, t - 1), 0)
                b_now = decision_variables["beet_flow_b"].get((i, 0, 0, t), 0) + decision_variables["beet_flow_b"].get(
                    (j, 0, 0, t), 0)
            if i == j:
                b_prev = decision_variables["beet_flow_b"].get((i, 0, 0, t - 1), 0)
                b_now = decision_variables["beet_flow_b"].get((i, 0, 0, t), 0)

            work_done = (b_prev - b_now)
            work_done_equivalent = work_done / (L_bar[l])
            metrics[l]['Moved_Beets'] += work_done
            metrics[l]['Actual_Work'] += work_done_equivalent  # includes partial work and work periods
    # Calculate Hidden Idle
    for l in machine_ids:
        metrics[l]['Hidden_Idle'] = metrics[l]['Onsite_Work_Periods'] + metrics[l]['Partial_Work_Travel'] - metrics[l][
            'Actual_Work']

    # Create df
    df_insights = pd.DataFrame.from_dict(metrics, orient='index')
    df_insights.index.name = 'Machine'

    # --- Perform the check on the DataFrame columns ---
    # Check: Hidden_Idle + Actual_Work == Onsite_Work_Periods + Partial_Work_Travel
    epsilon = 1e-4  # Tolerance for floating point comparison

    for machine_id, row in df_insights.iterrows():
        lhs_check = row['Hidden_Idle'] + row['Actual_Work']
        rhs_check = row['Onsite_Work_Periods'] + row['Partial_Work_Travel']

        # print(
        #    f"[DF CHECK for Machine {machine_id}] Hidden_Idle: {row['Hidden_Idle']:.4f}, "
        #    f"Actual_Work: {row['Actual_Work']:.4f}")
        # print(
        #    f"[DF CHECK for Machine {machine_id}] Onsite_Work_Periods: {row['Onsite_Work_Periods']:.4f},
        #    Partial_Work_Travel: {row['Partial_Work_Travel']:.4f}")
        print(f"[DF CHECK for Machine {machine_id}] LHS (Hidden_Idle + Actual_Work): {lhs_check:.4f}")
        print(f"[DF CHECK for Machine {machine_id}] RHS (Onsite_Work_Periods + Partial_Work_Travel): {rhs_check:.4f}")

        if abs(lhs_check - rhs_check) > epsilon:
            error_message = (
                f"DataFrame column check FAILED for Machine {machine_id}:\n"
                f"Hidden_Idle + Actual_Work != Onsite_Work_Periods + Partial_Work_Travel.\n"
                f"LHS ({row['Hidden_Idle']:.4f} + {row['Actual_Work']:.4f} = {lhs_check:.4f}) !=\n"
                f"RHS ({row['Onsite_Work_Periods']:.4f} + {row['Partial_Work_Travel']:.4f} = {rhs_check:.4f}).\n"
                f"Difference: {lhs_check - rhs_check:.4f}"
            )
            raise ValueError(error_message)
        else:
            print(f"[Hidden Idle DF CHECK for Machine {machine_id}] Balance check PASSED within tolerance {epsilon}.")

    # Reorder columns for better readability if desired
    column_order = [
        'Time_Periods', 'Depot_Periods', 'Onsite_Work_Periods', 'Onsite_Idle_Periods',
        'Travel_Periods', 'Actual_Travel', 'Partial_Work_Travel', 'Travel_Idle_Time',
        'Moved_Beets', 'Actual_Work', 'Hidden_Idle'
    ]
    # Filter out any columns not present in df_insights.columns, in case some metrics were not calculated (e.g. if machine_ids was empty)
    if not df_insights.empty:
        existing_columns = [col for col in column_order if col in df_insights.columns]
        df_insights = df_insights[existing_columns]
    else:
        # If df_insights is empty, return an empty DataFrame with expected columns (or handle as preferred)
        df_insights = pd.DataFrame(columns=column_order)

    return df_insights


def create_accounting_table(decision_variables, c_l, c_s_t, BEET_PRICE, UD_PENALTY, LAMBDA, operations_cost_t,
                            holidays=None):
    """
    Creates two outputs:
      1) A dictionary `accounting` with all key accounting figures
      2) A DataFrame `df_accounting` with rows for each cost or revenue item.

    It distinguishes Travel Costs, Partial Work Costs, Work Costs, Inventory Costs,
    as well as penalties for Unmet Demand and Idle time.
    """

    # --- 0) Basic sets ---
    T = {t for (_, _, _, t) in decision_variables['beet_flow_b'].keys()}
    I = {i for (i, _, _, _) in decision_variables['beet_flow_b'].keys()}
    L = list(c_l.keys())  # loader set from c_l dictionary

    # --- 1) Revenue ---
    max_t = max(T)
    revenue = 0.0
    for i in I:
        # from objective: (b[i,2,2,Tmax] + b[i,1,1,Tmax]) * BEET_PRICE
        revenue += (
                           decision_variables['beet_flow_b'][(i, 2, 2, max_t)]
                           + decision_variables['beet_flow_b'][(i, 1, 1, max_t)]
                   ) * BEET_PRICE

    # --- 2) Work, Partial, Travel Costs ---
    work_cost = 0.0
    partial_work_cost = 0.0
    travel_cost = 0.0

    # (a) Identify raw travel or working arcs
    for l in L:
        for (ll, i, j, t), val_x in decision_variables['loader_flow_x'].items():
            if ll != l or val_x < 1e-5:
                continue
            cost_ij = c_l[l].at[i, j]

            if i == j:
                # Working cost (if not idle)
                if decision_variables['working'].get((l, i, t), 0) > 0.5:
                    work_cost += cost_ij
            else:
                # i != j => cost_ij includes Travel + partial portion
                travel_cost += cost_ij

    # (b) Extract partial work from the arcs
    for l in L:
        # y_in
        for (ll, i, j, t), val_in in decision_variables.get('y_in', {}).items():
            if ll == l and i != j and val_in > 1e-5:
                partial_work_cost += operations_cost_t * val_in
                # print("Val in in accounting", (ll, i, j, t), val_in)
        # y_out
        for (ll, i, j, t), val_out in decision_variables.get('y_out', {}).items():
            if ll == l and i != j and val_out > 1e-5:
                partial_work_cost += operations_cost_t * val_out

                # print("Val out in accounting", (ll, i, j, t), val_out)
    # print("operations_cost_t", operations_cost_t)
    # (c) Subtract partial from the “raw” travel cost, to isolate real travel vs. partial
    travel_cost -= partial_work_cost

    # --- 3) Inventory Cost ---
    inventory_cost = 0.0
    for (i, s1, s2, t), b_val in decision_variables['beet_flow_b'].items():
        if (s1, s2) == (1, 1):
            inventory_cost += c_s_t * b_val

    # --- 4) Penalties (Unmet Demand + Idle) ---
    unmet_demand_penalty = 0.0
    if 'unmet_demand' in decision_variables:
        for t in T:
            ud = decision_variables['unmet_demand'].get(t, 0)
            unmet_demand_penalty += UD_PENALTY * ud

    idle_penalty = 0.0

    for l in L:
        for (ll, i, t), val_idle in decision_variables.get('idle', {}).items():
            if ll == l and val_idle > 0.5 and t > 0:
                if holidays:
                    if t not in holidays:
                        # Idle cost from objective: c_l[l].at[i,i] * LAMBDA
                        idle_penalty += c_l[l].at[i, i] * LAMBDA
                else:
                    # Idle cost from objective: c_l[l].at[i,i] * LAMBDA
                    idle_penalty += c_l[l].at[i, i] * LAMBDA

    # --- 5) Operating Profit and Objective Value ---
    operating_costs = work_cost + travel_cost + partial_work_cost + inventory_cost
    operating_profit = revenue - operating_costs
    total_penalties = unmet_demand_penalty + idle_penalty
    objective_value = operating_profit - total_penalties

    # --- 6) Build final table ---
    def ratio(x):
        return (x / revenue * 100) if revenue else 0.0

    # Dictionary form
    accounting = {
        "Revenue": (revenue, f"{ratio(revenue):.2f}%"),
        "Work Costs": (work_cost, f"{ratio(work_cost):.2f}%"),
        "Travel Costs": (travel_cost, f"{ratio(travel_cost):.2f}%"),
        "Partial Costs": (partial_work_cost, f"{ratio(partial_work_cost):.2f}%"),
        "Inventory Costs": (inventory_cost, f"{ratio(inventory_cost):.2f}%"),
        "Operating Profit": (operating_profit, f"{ratio(operating_profit):.2f}%"),
        "Penalties": {
            "Unmet Demand": (unmet_demand_penalty, f"{ratio(unmet_demand_penalty):.2f}%"),
            "Idle": (idle_penalty, f"{ratio(idle_penalty):.2f}%")
        },
        "Objective Value": (objective_value, f"{ratio(objective_value):.2f}%")
    }

    # DataFrame form
    # We also want a row for partial costs
    # Also combine all penalties in a "Total Penalties" row
    total_penalties_value = unmet_demand_penalty + idle_penalty

    data = {
        "Metric": [
            "Revenue",
            "Work Costs",
            "Partial Costs",
            "Travel Costs",
            "Inventory Costs",
            "Operating Profit",
            "Unmet Demand Penalty",
            "Idle Penalty",
            "Total Penalties",
            "Objective Value"
        ],
        "Value (€)": [
            revenue,
            work_cost,
            partial_work_cost,
            travel_cost,
            inventory_cost,
            operating_profit,
            unmet_demand_penalty,
            idle_penalty,
            total_penalties_value,
            objective_value
        ],
        "Ratio (%)": [
            ratio(revenue),
            ratio(work_cost),
            ratio(partial_work_cost),
            ratio(travel_cost),
            ratio(inventory_cost),
            ratio(operating_profit),
            ratio(unmet_demand_penalty),
            ratio(idle_penalty),
            ratio(total_penalties_value),
            ratio(objective_value)
        ]

    }

    df_accounting = pd.DataFrame(data)
    print(f"\nAccounting df with penalty ({UD_PENALTY}): ", df_accounting, "\n")
    return accounting, df_accounting


def create_excel_results(combined_key_list, experiments, base_file_path, flow_type, date=None):
    for experiment in experiments:

        # Load data based on experiment:
        # Construct the file path based on whether date is provided
        if date is not None:
            # Use filename with date
            results_file_path = os.path.join(base_file_path,
                                             f"results/reporting/results_{experiment}_{date}.pkl")
            print(f"Attempting to load dated file: {results_file_path}")  # Optional: for debugging
        else:
            # Use filename without date
            results_file_path = os.path.join(base_file_path,
                                             f"results/reporting/results_{experiment}.pkl")
            print(f"Attempting to load non-dated file: {results_file_path}")  # Optional: for debugging

        # Loop over combined_key list to access data
        for combined_key in combined_key_list:
            results = load_results(results_file_path)

            # Extract parameters and decision variables
            print(results.keys())
            print(results[combined_key].keys())
            if results[combined_key].get("opt_type") == "time_agg":
                print("Process Time Agg Excel Results")
                parameters = results[combined_key]["scenario_results_fine"]["parameters"]
                decision_variables = results[combined_key]["scenario_results_fine"]["decision_variables"]
            else:
                parameters = results[combined_key]["parameters"]
                decision_variables = results[combined_key]["decision_variables"]

            # print("beet_flow : ", results[combined_key]["decision_variables"]['beet_flow_b'].keys())
            # Extract parameters

            if flow_type == "full_flow":
                c_h = parameters["c_h"]
                H = parameters["H"]

            c_l = parameters["c_l"]
            L = parameters["L"]
            sugar_price = parameters["sugar_price"]
            sugar_concentration = parameters["sugar_concentration"]

            tau = parameters.get("tau")
            L_bar = parameters.get("L_bar")

            # Extract accounting parameters with defaults
            c_s_t = parameters.get("c_s_t", 0.02)
            BEET_PRICE = parameters.get("BEET_PRICE", 1.5)
            UD_PENALTY = parameters.get("UD_PENALTY", 3)
            LAMBDA = parameters.get("LAMBDA", 0.75)
            operations_cost_t = parameters.get("operations_cost_t", 230)
            holidays = parameters.get("holidays", None)

            # Save results to Excel
            unique_key = f"{experiment}_{combined_key}"

            results = {}

            # Machine Schedule
            if flow_type == "full_flow":
                machinery_types = ['harvester_flow_x', 'loader_flow_x']
            if flow_type == "load_flow":
                machinery_types = ['loader_flow_x']

            results['df_schedule'] = extract_machinery_schedule_idle_twolevel(decision_variables)

            # Get an arbitrary element from the set L
            any_loader = next(iter(L))

            # Machinery Operations Costs
            results['df_loader_cost'] = extract_machine_costs_with_partial(
                decision_variables=decision_variables,
                machinery_type='loader_flow_x',
                cost_matrix=c_l,
                machinery_set=L,
                op_cost=c_l[any_loader].iloc[1, 1]
            )

            if flow_type == "full_flow":
                results['df_harvest_cost'] = extract_machine_costs_with_partial(
                    decision_variables=decision_variables,
                    machinery_type='harvester_flow_h',
                    cost_matrix=c_h,
                    machinery_set=H,
                    op_cost=c_h.iloc[1, 1]
                )

                results['df_machinery_cost'] = pd.concat([results['df_harvest_cost'], results['df_loader_cost']],
                                                         axis=0)

            if flow_type == "load_flow":
                results['df_machinery_cost'] = results['df_loader_cost']

            # Machinery Cost Matrix
            if flow_type == "full_flow":
                results['df_harvester_cost_matrix'] = create_machine_cost_parameter_df(c_h)
            # results['df_loader_cost_matrix'] = create_machine_cost_parameter_df(c_l)

            # Beet Movement
            results['df_beet_movement'] = beet_movement_info(decision_variables, flow_type=flow_type)

            # Beet yields per field
            results['df_field_yield'] = extract_field_yield(decision_variables)

            # Revenue Unmet Demand
            results['revenue_unmet_df'] = calculate_revenue_and_unmet_demand(
                decision_variables, sugar_price, sugar_concentration, flow_type=flow_type)

            # Create accounting table and add to results
            accounting, df_accounting = create_accounting_table(
                decision_variables=decision_variables,
                c_l=c_l,
                c_s_t=c_s_t,
                BEET_PRICE=BEET_PRICE,
                UD_PENALTY=UD_PENALTY,
                LAMBDA=LAMBDA,
                operations_cost_t=operations_cost_t,
                holidays=holidays
            )
            results['df_accounting'] = df_accounting

            #print("LBAR in Excel Results: ", L_bar)

            if tau is not None and L_bar is not None:
                results['df_hidden_idle_insights'] = create_hidden_idle_insights_table(decision_variables, tau, L_bar)

            with pd.ExcelWriter(f'{base_file_path}results/excel/{unique_key}_flow_results.xlsx') as writer:

                results['df_beet_movement'].to_excel(writer, sheet_name='Beet Movement')
                results['df_schedule'].to_excel(writer, sheet_name='Machine Schedule')
                results['df_field_yield'].to_excel(writer, sheet_name='Field Yield')
                results['df_machinery_cost'].to_excel(writer, sheet_name='Machinery Cost')
                if flow_type == "full_flow":
                    results['df_harvester_cost_matrix'].to_excel(writer, sheet_name='c_h')
                results['revenue_unmet_df'].to_excel(writer, sheet_name='Revenue and Unmet Demand')
                results['df_accounting'].to_excel(writer, sheet_name='Accounting')
                if 'df_hidden_idle_insights' in results:
                    results['df_hidden_idle_insights'].to_excel(writer, sheet_name='Hidden Idle')


def post_process_decision_variables(decision_variables, verbose=None):
    # Determine the cutoff time period
    T = max(t for _, _, _, t in decision_variables['beet_flow_b'].keys())
    I = max(i for i, _, _, _ in decision_variables['beet_flow_b'].keys())

    cutoff_time = T
    for t in range(T + 1):
        sum_beet_flow = sum(
            decision_variables['beet_flow_b'][i, 0, 0, t] +
            decision_variables['beet_flow_b'][i, 1, 1, t] +
            decision_variables['beet_flow_b'][i, 2, 2, t]
            for i in range(1, I + 1)
        )
        if sum_beet_flow <= 0.01:
            cutoff_time = t
            if verbose:
                print("Cutoff at:", cutoff_time)
                print(f"We cutted of {T - cutoff_time} periods")
            break

    # Truncate decision variables

    # Truncate decision variables
    truncated_decision_variables = {}
    for key, var_dict in decision_variables.items():
        if key == 'unmet_demand':
            truncated_decision_variables[key] = {k: v for k, v in var_dict.items() if k <= cutoff_time}
        else:
            truncated_decision_variables[key] = {k: v for k, v in var_dict.items() if k[-1] <= cutoff_time}

    return truncated_decision_variables


# Define a function to calculate the metrics for a given result
def calculate_metrics(result):
    total_costs = result["df_machinery_cost"].sum().sum()
    inventory_sum = result["df_beet_movement"].loc["Inventory_t"].sum()
    unmet_demand_periods = (result["revenue_unmet_df"].loc["UD_t"] > 0).sum()
    unmet_demand_sum = (result["revenue_unmet_df"].loc["UD_t"]).sum()
    unmet_demand_var = (result["revenue_unmet_df"].loc["UD_t"]).var()
    loaded_sum = result["df_beet_movement"].loc["Loaded_t"].sum()

    beet_sum = (
            result["df_beet_movement"].loc["Total_Loaded"].iloc[-1] +
            result["df_beet_movement"].loc["Inventory_t"].iloc[-1]
    )

    return total_costs, beet_sum, inventory_sum, unmet_demand_periods, unmet_demand_sum, unmet_demand_var, loaded_sum


def create_kpi_comparison(gurobi_results, heuristic_results, combined_key_list, flow_type):
    # Initialize a list to store comparison data for each scenario
    comparison_data = []
    # Iterate over all scenarios and calculate the required metrics
    for scenario in combined_key_list:

        if gurobi_results[scenario].get("opt_type") == "time_agg":
            print("Process Time Agg KPI's")
            time_agg = True
            gurobi_parameters = gurobi_results[scenario]["scenario_results_fine"]["parameters"]
            gurobi_parameters_coarse = gurobi_results[scenario]["scenario_results_coarse"]["parameters"]
            gurobi_decision_variables = gurobi_results[scenario]["scenario_results_fine"]["decision_variables"]
            heuristic_decision_variables = heuristic_results[scenario]["decision_variables"]

        else:
            print("Process No Time Agg KPI's")
            time_agg = False
            # Extract Parameters and Decision Variables
            gurobi_parameters = gurobi_results[scenario]["parameters"]
            heuristic_parameters = heuristic_results[scenario]["parameters"]
            gurobi_decision_variables = gurobi_results[scenario]["decision_variables"]
            heuristic_decision_variables = heuristic_results[scenario]["decision_variables"]

        # Extract parameters
        if flow_type == "full_flow":
            c_h = gurobi_parameters["c_h"]
            H = gurobi_parameters["H"]
        else:
            c_h = None
            H = None

        c_l = gurobi_parameters["c_l"]
        L = gurobi_parameters["L"]
        T = gurobi_parameters["T"]
        t_p = gurobi_parameters["t_p"]
        days = (len(T) * t_p) / 24

        gurobi_MIP_gap = np.round(gurobi_parameters["MIP_gap"] * 100, 4)
        runtime = gurobi_parameters["runtime"]

        if time_agg:
            gurobi_MIP_gap_coarse = np.round(gurobi_parameters_coarse["MIP_gap"] * 100, 4)
            runtime_coarse = gurobi_parameters_coarse["runtime"]

        sugar_price = gurobi_parameters["sugar_price"]
        sugar_concentration = gurobi_parameters["sugar_concentration"]

        mcf_result = create_all_scheduling_results(
            gurobi_decision_variables, c_l, L, sugar_price, sugar_concentration,
            flow_type=flow_type, c_h=c_h, H=H)

        heuristic_result = create_all_scheduling_results(
            heuristic_decision_variables, c_l, L, sugar_price, sugar_concentration,
            flow_type=flow_type, c_h=c_h, H=H)

        # Heuristic
        h_tc, h_beet_sum, h_inventory_sum, h_ud_periods, h_ud_sum, h_ud_var, h_loaded_sum = \
            calculate_metrics(heuristic_result)

        # MCF
        mcf_tc, mcf_beet_sum, mcf_inventory_sum, mcf_ud_periods, mcf_ud_sum, mcf_ud_var, mcf_loaded_sum \
            = calculate_metrics(mcf_result)

        # Calc Revenue
        BEET_PRICE = gurobi_parameters["BEET_PRICE"]
        h_rev = h_beet_sum * BEET_PRICE
        mcf_rev = mcf_beet_sum * BEET_PRICE

        # Calculate Logistical Margin
        h_margin = np.round((h_tc / h_rev) * 100, 2)
        mcf_margin = np.round((mcf_tc / mcf_rev) * 100, 2)

        # Calculate relative values from the perspective of MCF
        cost_diff = mcf_tc - h_tc
        inventory_diff = mcf_inventory_sum - h_inventory_sum
        unmet_demand_s_diff = mcf_ud_sum - h_ud_sum

        cost_rel_diff = (cost_diff / h_tc) * 100 if h_tc != 0 else float('inf')

        kpi_row = {
            "Scen": scenario,  # Scenario
            "MCF_TC": mcf_tc,  # MCF Total Costs
            "Heur_TC": h_tc,  # Heuristic Total Costs
            "CostDiff_%": cost_rel_diff,  # Cost Relative Difference (%)
            "Inv_s_D": inventory_diff,
            "MCF_Rev": mcf_rev,  # MCF Revenue
            "Heur_Rev": h_rev,  # Heuristic Revenue
            "H_Margin": h_margin,
            "MCF_Margin": mcf_margin,
            "UD_s_D": unmet_demand_s_diff,  # Unmet Demand Sum Difference
            "MCF_Beets": mcf_loaded_sum,
            "Heur_Beets": h_loaded_sum,
            "Days": days,
            "MCF_GAP": gurobi_MIP_gap,
            "Time": runtime,
        }

        if time_agg:
            kpi_row["MCF_GAP_coarse"] = gurobi_MIP_gap_coarse
            kpi_row["Time_coarse"] = runtime_coarse

        comparison_data.append(kpi_row)

    # Create a DataFrame from the comparison data
    comparison_df = pd.DataFrame(comparison_data)

    # Print the comparison DataFrame
    return comparison_df


def create_combined_key_list(scenarios, gurobi_model_versions, sensitivity_scenarios):
    combined_key_list = []
    for scenario in scenarios:

        # Extract a unique scenario identifier (e.g., using fields, harvesters, and loaders)
        scenario_id = f"{scenario['nr_fields']}_{scenario['nr_h']}_{scenario['nr_l']}"

        count = 0

        for model_version in gurobi_model_versions:
            # Improved Key Naming
            model_version_key = f"MV_{int(model_version['idle_time'])}_{int(model_version['restricted'])}_" \
                                f"{int(model_version['access'])}_{int(model_version['working_hours'])}_" \
                                f"{int(model_version['travel_time'])}_{model_version['t_p']}"

            for sens_scenario in sensitivity_scenarios:
                # Improved Sensitivity ID Naming
                sensitivity_id = f"S_{sens_scenario['cost_multi']}_{sens_scenario['productivity_multi']}_" \
                                 f"{sens_scenario['working_hours_multi']}"

                # Unique identifier for storing results
                combined_key = f"{scenario_id}_{model_version_key}_{sensitivity_id}"

                combined_key_list.append(combined_key)

    return (combined_key_list)
