import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

# Use single results file
from src.mcf_utils.single_results_utils import (
    create_hidden_idle_insights_table,
    extract_machine_costs_with_partial,
    extract_machinery_schedule_idle_twolevel,
    extract_machine_costs,
    create_machine_cost_parameter_df,
    beet_movement_info,
    calculate_revenue_and_unmet_demand,
    create_accounting_table,
    extract_field_yield
)


# TODO: Consolidate single and rolling result file

def load_results(file_path):
    with open(file_path, 'rb') as result_file:
        results = pickle.load(result_file)
    return results


def create_all_scheduling_results(decision_variables, parameters, flow_type, c_h=None, H=None):
    results = {}

    # Machine Schedule
    machinery_types = ['harvester_flow_x', 'loader_flow_x']
    results['df_schedule'] = extract_machinery_schedule_idle_twolevel(decision_variables)

    # Extract Parameters
    c_l = parameters["c_l"]
    L = parameters["L"]

    sugar_price = parameters["sugar_price"]
    sugar_concentration = parameters["sugar_concentration"]

    # --- NEW: Extract accounting parameters with defaults ---
    c_s_t = parameters.get("c_s_t", 0.02)
    BEET_PRICE = parameters.get("BEET_PRICE", 1.5)
    UD_PENALTY = parameters.get("UD_PENALTY", 3)
    LAMBDA = parameters.get("LAMBDA", 0.75)
    operations_cost_t = parameters.get("operations_cost_t", 230)
    tau = parameters.get("tau")
    L_bar = parameters.get("L_bar")
    holidays = parameters.get("holidays", None)

    # Cost
    # TODO: adapt cost calc
    results['df_loader_cost'] = extract_machine_costs_with_partial(
        decision_variables=decision_variables,
        machinery_type='loader_flow_x',
        cost_matrix=c_l,
        machinery_set=L,
        op_cost=operations_cost_t
    )

    if c_h is not None:
        results['df_harvest_cost'] = extract_machine_costs(decision_variables, 'harvester_flow_x', c_h, H)
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

    # Accounting
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

    if tau is not None and L_bar is not None:
        results['df_hidden_idle_insights'] = create_hidden_idle_insights_table(decision_variables, tau, L_bar)

    return results



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


def create_kpi_comparison(gurobi_results, heuristic_results, flow_type):
    # Initialize a list to store comparison data for each scenario
    comparison_data = []

    if gurobi_results.get("opt_type") == "time_agg":
        print("Process Time Agg KPI's")
        time_agg = True
        gurobi_parameters = gurobi_results["scenario_results_fine"]["parameters"]
        gurobi_parameters_coarse = gurobi_results["scenario_results_coarse"]["parameters"]
        gurobi_decision_variables = gurobi_results["scenario_results_fine"]["decision_variables"]
        heuristic_parameters = heuristic_results["parameters"]
        heuristic_decision_variables = heuristic_results["decision_variables"]

    else:
        print("Process No Time Agg KPI's")
        time_agg = False
        # Extract Parameters and Decision Variables
        gurobi_parameters = gurobi_results["parameters"]
        heuristic_parameters = heuristic_results["parameters"]
        gurobi_decision_variables = gurobi_results["decision_variables"]
        heuristic_decision_variables = heuristic_results["decision_variables"]

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

    mcf_result = create_all_scheduling_results(
        decision_variables=gurobi_decision_variables,
        parameters=gurobi_parameters,
        flow_type=flow_type,
        c_h=None,
        H=None)

    heuristic_result = create_all_scheduling_results(
        decision_variables=heuristic_decision_variables,
        parameters=heuristic_parameters,
        flow_type=flow_type,
        c_h=None,
        H=None)

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

    return comparison_df


def create_consolidated_excel_with_kpi_comparison(
        pickle_file_path,
        base_file_path,
        flow_type,
        gap_rows=2,  # Number of empty rows as gaps between sections
        group_id=None,
        file_name="rolling"
):
    """
    Creates a consolidated Excel file with all weekly results, handling Gurobi vs. Heuristic,
    plus KPI comparisons, with multi-index columns preserved (so top row merges for e.g. (time, start_pos)).
    """

    # 1) Load the pickle file
    try:
        with open(pickle_file_path, 'rb') as f:
            weekly_results = pickle.load(f)
    except FileNotFoundError:
        print(f"Pickle file not found at: {pickle_file_path}")
        return
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return

    # 2) Possibly filter by group_id
    if group_id is not None:
        try:
            weekly_results = weekly_results[group_id].copy()
        except KeyError:
            print(f"Group ID {group_id} not found in the weekly results.")
            return

    # 3) Check for KPI comparisons
    include_kpi = any('kpi_comparison' in week_data for week_data in weekly_results.values())

    # 4) Sort weeks, e.g. Week_1, Week_2, ...
    try:
        sorted_weeks = sorted(
            weekly_results.keys(),
            key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0
        )
    except Exception as e:
        print(f"Error sorting weeks: {e}")
        return

    # 5) Prepare the output Excel
    output_excel_path = os.path.join(base_file_path, f'consolidated_weekly_results_{file_name}_{group_id}.xlsx')
    os.makedirs(base_file_path, exist_ok=True)

    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Optional: define a header format (bold + background)
        header_format = workbook.add_format({'bold': True, 'bg_color': '#dcdde0'})

        # Create worksheets for each metric
        metrics_list = [
            'Beet Movement',
            'Machine Schedule',
            'Beet Yield',
            'Revenue and Unmet Demand',
            'Loader Cost',
            'Accounting',
            'Idle',
            'KPI Comparison'  # Will only be used if include_kpi is True
        ]
        worksheets = {}
        current_rows = {}
        for metric in metrics_list:
            worksheets[metric] = workbook.add_worksheet(metric)
            writer.sheets[metric] = worksheets[metric]
            current_rows[metric] = 0

        # 6) Iterate over each Week
        for week in sorted_weeks:
            print(f"Processing {week}...")
            week_data = weekly_results[week]

            gurobi_results_obj = week_data.get('gurobi_results', {})
            heuristic_results_obj = week_data.get('gurobi_heuristic_results_pl', {})
            kpi_comparison = week_data.get('kpi_comparison', pd.DataFrame())

            # Extract the scheduling results
            try:
                is_time_agg = gurobi_results_obj.get("opt_type") == "time_agg"
                if is_time_agg:
                    # Get the final 1h solution for detailed operational reports
                    gurobi_final_solution = gurobi_results_obj.get('scenario_results_fine', {})
                    gurobi_decision_variables = gurobi_final_solution.get('decision_variables', {})
                    gurobi_parameters = gurobi_final_solution.get('parameters', {})

                    heuristic_decision_variables = heuristic_results_obj.get('decision_variables', {})
                    heuristic_parameters = heuristic_results_obj.get('parameters', {})

                else:
                    # Fallback for old, non-time-agg results
                    gurobi_decision_variables = gurobi_results_obj.get('decision_variables', {})
                    gurobi_parameters = gurobi_results_obj.get('parameters', {})

                    heuristic_decision_variables = heuristic_results_obj.get('decision_variables', {})
                    heuristic_parameters = heuristic_results_obj.get('parameters', {})

                # The rest of the function uses these variables, so it now works for both cases
                gurobi_scheduling_results = create_all_scheduling_results(
                    decision_variables=gurobi_decision_variables,
                    parameters=gurobi_parameters,
                    flow_type=flow_type,
                    c_h=None, H=None
                )

                # The rest of the function uses these variables, so it now works for both cases
                heuristic_scheduling_results = create_all_scheduling_results(
                    decision_variables=heuristic_decision_variables,
                    parameters=heuristic_parameters,
                    flow_type=flow_type,
                    c_h=None, H=None
                )

            except Exception as e:
                print(f"Error creating scheduling results for {week}: {e}")
                continue

            # 7) Extract each DataFrame
            gurobi_df_beet_movement = gurobi_scheduling_results.get('df_beet_movement', pd.DataFrame())
            gurobi_df_schedule = gurobi_scheduling_results.get('df_schedule', pd.DataFrame())
            gurobi_df_field_yield = gurobi_scheduling_results.get('df_field_yield', pd.DataFrame())
            gurobi_revenue_unmet_df = gurobi_scheduling_results.get('revenue_unmet_df', pd.DataFrame())
            gurobi_df_loader_cost = gurobi_scheduling_results.get('df_loader_cost', pd.DataFrame())
            gurobi_df_accounting = gurobi_scheduling_results.get('df_accounting', pd.DataFrame())
            gurobi_df_idle = gurobi_scheduling_results.get('df_hidden_idle_insights', pd.DataFrame())

            heuristic_df_beet_movement = heuristic_scheduling_results.get('df_beet_movement', pd.DataFrame())
            heuristic_df_schedule = heuristic_scheduling_results.get('df_schedule', pd.DataFrame())
            heuristic_df_field_yield = heuristic_scheduling_results.get('df_field_yield', pd.DataFrame())
            heuristic_revenue_unmet_df = heuristic_scheduling_results.get('revenue_unmet_df', pd.DataFrame())
            heuristic_df_loader_cost = heuristic_scheduling_results.get('df_loader_cost', pd.DataFrame())
            heuristic_df_accounting = heuristic_scheduling_results.get('df_accounting', pd.DataFrame())
            heuristic_df_idle = heuristic_scheduling_results.get('df_hidden_idle_insights', pd.DataFrame())

            # 8) Loop over the metric pairs
            metric_pairs = [
                ('Beet Movement', gurobi_df_beet_movement, heuristic_df_beet_movement),
                ('Machine Schedule', gurobi_df_schedule, heuristic_df_schedule),
                ('Beet Yield', gurobi_df_field_yield, heuristic_df_field_yield),
                ('Revenue and Unmet Demand', gurobi_revenue_unmet_df, heuristic_revenue_unmet_df),
                ('Loader Cost', gurobi_df_loader_cost, heuristic_df_loader_cost),
                ('Accounting', gurobi_df_accounting, heuristic_df_accounting),
                ('Idle', gurobi_df_idle, heuristic_df_idle),
            ]

            for metric_name, gurobi_df, heuristic_df in metric_pairs:
                worksheet = worksheets[metric_name]
                current_row = current_rows[metric_name]

                # ---- A) Gurobi section ----
                # Write a bold headline in the first column
                gurobi_headline = f"Gurobi Results - {week}"
                worksheet.write(current_row, 0, gurobi_headline, header_format)
                current_row += 1

                # Write the Gurobi DataFrame with multi-index columns intact
                if not gurobi_df.empty:
                    gurobi_df.to_excel(
                        writer,
                        sheet_name=metric_name,
                        startrow=current_row,
                        startcol=0,
                        index=True,
                        header=True,
                        merge_cells=True  # ensures multi-index merges
                    )
                    # Figure out how many header rows exist: multi-index columns => columns.nlevels
                    # So total height = data rows + number of header rows
                    df_height = len(gurobi_df) + gurobi_df.columns.nlevels
                    current_row += df_height
                else:
                    current_row += 1

                # Insert blank rows after Gurobi table
                for _ in range(gap_rows):
                    current_row += 1

                # ---- B) Heuristic section ----
                heuristic_headline = f"Gurobi Heuristic Results - {week}"
                worksheet.write(current_row, 0, heuristic_headline, header_format)
                current_row += 1

                if not heuristic_df.empty:
                    heuristic_df.to_excel(
                        writer,
                        sheet_name=metric_name,
                        startrow=current_row,
                        startcol=0,
                        index=True,
                        header=True,
                        merge_cells=True
                    )
                    df_height = len(heuristic_df) + heuristic_df.columns.nlevels
                    current_row += df_height
                else:
                    current_row += 1

                # Insert blank rows after Heuristic table
                for _ in range(gap_rows):
                    current_row += 1

                current_rows[metric_name] = current_row

            # 9) KPI Comparison
            if include_kpi and not kpi_comparison.empty:
                worksheet = worksheets['KPI Comparison']
                current_row = current_rows['KPI Comparison']

                # Headline
                kpi_headline = f"KPI Comparison - {week}"
                worksheet.write(current_row, 0, kpi_headline, header_format)
                current_row += 1

                # Write KPI DataFrame
                kpi_comparison.to_excel(
                    writer,
                    sheet_name='KPI Comparison',
                    startrow=current_row,
                    startcol=0,
                    index=True,
                    header=True,
                    merge_cells=True
                )
                # height = data rows + multi-index header rows (if any)
                df_height = len(kpi_comparison) + kpi_comparison.columns.nlevels
                current_row += df_height

                # Gap rows
                for _ in range(gap_rows):
                    current_row += 1

                current_rows['KPI Comparison'] = current_row

    print(f"Consolidated Excel file with KPI comparisons created at {output_excel_path}")
