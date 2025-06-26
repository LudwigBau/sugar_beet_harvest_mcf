import math
from datetime import datetime
# from src.mcf_utils.heuristic_utils import create_production_plan_schedule_tau
from src.mcf_utils.heuristic_utils import create_production_plan_schedule_tau
from src.mcf_utils.RealBenchRoutingClass import *
from src.mcf_utils.BeetFlowClass import BeetFlow
from src.mcf_utils.LoaderFlowClass import LoaderFlow
from src.mcf_utils.single_results_utils import *

from gurobipy import Model, GRB, quicksum
import gurobipy as gp

from collections import defaultdict
from typing import TypeAlias, Set, Dict, List, Tuple, Any, Optional

# Define Arc type alias for clarity
Arc: TypeAlias = Tuple[int, int]


def loading_flow_set_up_check(raw_instance, derived):
    T = derived["T"]

    # check inventory_goals and vehicle capacity limits to be matched
    # TODO: create set up check


def loading_flow_solution_check(params, decision_variables):
    # Select params
    L = params["L"]
    Il = params["Il"]
    I = params["I"]
    T = params["T"]
    tau = params["tau"]
    holidays = params["holidays"]
    epsilon = 1e-5

    # select variables
    beet_flow = decision_variables["beet_flow_b"]
    loader_flow = decision_variables["loader_flow_x"]
    finished = decision_variables["finished"]
    started = decision_variables["started"]
    idle = decision_variables["idle"]

    print("\nStart Post-Solution Checks...\n")

    # 1. check finished variable (for all beets at max (t) = must result in finished = 1)
    print("\n1. Check Finished:\n")
    finished_fields = []
    for (i, r1, r2, t), value in beet_flow.items():
        if t == 36:
            if r1 == 0:
                if r2 == 0:
                    if value == 0:
                        # optional print
                        # print((i, r1, r2, t), value)
                        finished_fields.append(i)

    print("Actual finished fields: \n", finished_fields, "Length: ", len(finished_fields))

    missmatch = []
    for (l, i, t), value in finished.items():
        if i in finished_fields:
            # optional print
            # print((l, i, t), value)
            if value <= epsilon:
                missmatch.append(i)

    if len(missmatch) > len(L):
        print(
            f"WARNING: More mismatched 'finished' indicators detected than number of loaders "
            f" for {len(missmatch)} empty fields.")
    else:
        print(f"Restricted works as expected. Less than len(L): {len(L)} variables are restricted")

    # 2. check all started (if any inflow on j -> started = 1)
    print("\n2. Check Started:\n")
    started_fields = []
    for (i, r1, r2, t), value in beet_flow.items():
        if t == max(T):
            if r1 == 0:
                if r2 == 0:
                    if value < beet_flow[(i, 0, 0, 0)] - 1:
                        # optional print
                        # print((i, r1, r2, t), value, beet_flow[(i, 0, 0, 0)])
                        started_fields.append(i)

    print("Actual started fields: \n", started_fields, "Length: ", len(started_fields))

    missmatch = []
    for (l, i, t), value in started.items():
        if i in started_fields:
            # optional print
            # print((l, i, t), value)
            if value <= epsilon:
                missmatch.append(i)

    if len(missmatch) > 0:
        print(
            f"WARNING: {len(missmatch)} number of mismatched 'started' indicators.")
        print("Started missmatch", missmatch, "Length: ")
    else:
        print(f"Started works as expected. {len(L)} variables are missmatched")

    # 3. check holiday for all t in Holiday we need all idle_i = 1 if x_i_i = 1
    print("\n3. Check Holiday:\n")
    holiday_violation = {}

    for l in L:
        holiday_violation[l] = []
        for t in holidays:
            for i in Il[l]:
                for j in Il[l]:
                    if (l, i, j, t) in loader_flow:
                        if loader_flow[l, i, j, t] > 0.5:
                            if i != j:
                                holiday_violation[l].append(t)
                            else:
                                # optional print
                                # print([l, i, j, t], loader_flow[l, i, j, t])
                                if (l, i, t) in idle:
                                    if idle == 0:
                                        holiday_violation[l].append(t)
                                else:
                                    holiday_violation[l].append(t)

    for l in L:
        if holiday_violation[l]:
            print(f"WARNING: Loader {l} violates holdiay constraint at t: {holiday_violation[l]}")
        else:
            print(f"Loader {l} respects holiday constraints")

    # 4. Check partial beet flow. if x_ij with i != j, use tau to check if sum of beet flow at b_i and b_j is less equal
    # 1 - tau. If true: y is not over allowing flow. Furthermore, check if any partial flow uses the max flow to check
    # if max potential is used. If not: indication for a problem (why would the model use full potential at least once?)
    # print("\n4. Check Partial Work:")

    # TODO: add partial work check


def extract_last_period_beet_volume(
        params: dict,
        decision_variables: dict
) -> dict:

    last_period_beet_volume = {}

    # Extract variables and params from Gurobi solution
    beets = decision_variables['beet_flow_b']
    last_time_period = max(params["T"])
    fields = params["I"]

    # Fill empty dict with last time period beet volumes at field (0,0) position
    for i in fields:
        last_period_beet_volume[i] = beets[i, 0, 0, last_time_period]

    return last_period_beet_volume


def production_plan_holiday_to_t(df, periods_per_day=24 // 2):
    final_periods = []

    # Iterate over each row in the DataFrame
    for day_index, row in df.iterrows():
        if row['Holiday']:
            # Calculate the start and end period indices for this day
            # Shift to the right to account for setup period
            start_period = day_index * periods_per_day + 1
            end_period = start_period + periods_per_day  # Exclusive

            # Extend the holiday_periods list with periods for this day
            final_periods.extend(range(start_period, end_period))

    return final_periods


def enforce_schedule_with_constraints_idle(model, machinery_flow, schedules, Ih, T):
    # Add constraints for all potential movements
    for h in schedules.keys():
        schedule_len = len(schedules[h])

        for t in T:
            for i in Ih[h]:
                for j in Ih[h]:
                    if t > 0:
                        # Inside the schedule time range
                        if t < schedule_len - 1:
                            # idle, working only present when staying at a position (i==j)
                            if i == j:
                                if [str(i), str(j)] == schedules[h][t]:  # Enforce scheduled movement or operation
                                    model.addConstr(machinery_flow.idle[h, i, t] == 1,
                                                    f"idle_schedule_x_{h}_{i}_{j}_{t}")
                                elif [i, j] == schedules[h][t]:
                                    # Enforce that the machine is working in place
                                    model.addConstr(machinery_flow.working[h, i, t] == 1,
                                                    f"working_schedule_x_{h}_{i}_{j}_{t}")

                            # Movements are unaffected by idle dynamic
                            if i != j:
                                if [i, j] == schedules[h][t]:
                                    # Constrain this movement to 1 as per the schedule
                                    model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                    f"schedule_x_{h}_{i}_{j}_{t}")

                                elif [str(i), j] == schedules[h][t]:
                                    # Special case
                                    model.addConstr(machinery_flow.x[h, i, i, t] == 1,
                                                    f"schedule_x_{h}_{i}_{j}_{t}")

                                else:
                                    # Constrain all non-scheduled movements at this time to 0
                                    model.addConstr(machinery_flow.x[h, i, j, t] == 0,
                                                    f"no_schedule_x_{h}_{i}_{j}_{t}")

                        # Last schedule step
                        elif t == schedule_len - 1:
                            # Add the last step in the schedule
                            if [i, j] == schedules[h][t]:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                f"end_x_{h}_{i}_{j}_{t}")
                            else:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 0,
                                                f"noend_x_{h}_{i}_{j}_{t}")
                        # Post Schedule
                        elif t > schedule_len - 1:
                            # After the schedule ends, move to position 0
                            if i == schedules[h][schedule_len - 1][1] and j == 0:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                f"postschedule_x_{h}_{i}_{j}_{t}")
                            elif i == 0 and j == 0:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                f"stay_x_{h}_{i}_{j}_{t}")
                            else:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 0,
                                                f"nopostschedule_x_{h}_{i}_{j}_{t}")

    model.update()


def from_schedule_to_hot_start_idle(model, machinery_flow, schedules, Ih, T):
    # Set starting values for all potential movements
    for h in schedules.keys():
        schedule_len = len(schedules[h])

        for t in T:
            for i in Ih[h]:
                for j in Ih[h]:
                    if t > 0:
                        # Inside the schedule time range
                        if t < schedule_len - 1:
                            # idle, working only present when staying at a position (i==j)
                            if i == j:
                                if [str(i), str(j)] == schedules[h][t]:  # Enforce scheduled movement or operation
                                    machinery_flow.idle[h, i, t].start = 1

                                elif [i, j] == schedules[h][t]:
                                    # Enforce that the machine is working in place
                                    machinery_flow.working[h, i, t].start = 1

                            # Movement are unaffected by idle dynamic
                            if i != j:
                                if [i, j] == schedules[h][t]:
                                    # Constrain this movement to 1 as per the schedule
                                    machinery_flow.x[h, i, j, t].start = 1
                                elif [str(i), j] == schedules[h][t]:
                                    # Constrain this movement to 1 as per the schedule
                                    machinery_flow.x[h, i, i, t].start = 1

                                else:
                                    # Constrain all non-scheduled movements at this time to 0
                                    machinery_flow.x[h, i, j, t].start = 0

                        elif t == schedule_len - 1:
                            # At the last step in the schedule
                            if [i, j] == schedules[h][t]:
                                machinery_flow.x[h, i, j, t].start = 1
                            else:
                                machinery_flow.x[h, i, j, t].start = 0
                        elif t > schedule_len - 1:
                            # After the schedule ends, move to position 0
                            if i == schedules[h][schedule_len - 1][1] and j == 0:
                                machinery_flow.x[h, i, j, t].start = 1
                            elif i == 0 and j == 0:
                                machinery_flow.x[h, i, j, t].start = 1
                            else:
                                machinery_flow.x[h, i, j, t].start = 0

    model.update()


def enforce_schedule_with_constraints_pruned(model, machinery_flow, schedules, Ih, T):
    """
    Enforce a pre-computed schedule as hard constraints. The schedule (dict)
    defines for each machine h a list of moves [ [i,j], ... ] for each time t.
    This version handles idle/working variables and is adapted to use only valid
    arcs (i.e. those present in machinery_flow.x).
    """
    for h in schedules.keys():
        schedule_len = len(schedules[h])
        for t in T:
            for i in Ih[h]:
                for j in Ih[h]:
                    # Check if the arc (h, i, j, t) exists in the pruned index set.
                    if (h, i, j, t) not in machinery_flow.x:
                        continue  # skip if not a valid combination.

                    if t > 0:
                        if t < schedule_len - 1:
                            # For self-loops: enforce idle/working decisions.
                            if i == j:
                                # In some cases the scheduled move is stored as strings.
                                if [str(i), str(j)] == schedules[h][t]:
                                    model.addConstr(machinery_flow.idle[h, i, t] == 1,
                                                    f"idle_schedule_x_{h}_{i}_{j}_{t}")
                                elif [i, j] == schedules[h][t]:
                                    model.addConstr(machinery_flow.working[h, i, t] == 1,
                                                    f"working_schedule_x_{h}_{i}_{j}_{t}")
                            # For moves (i != j), force the scheduled move.
                            if i != j:
                                if [i, j] == schedules[h][t]:
                                    model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                    f"schedule_x_{h}_{i}_{j}_{t}")
                                elif [str(i), j] == schedules[h][t]:
                                    print(f"Warning idle movement! from str {i} to int {j}")
                                    model.addConstr(machinery_flow.x[h, i, i, t] == 1,
                                                    f"schedule_x_{h}_{i}_{i}_{t}")
                                else:
                                    model.addConstr(machinery_flow.x[h, i, j, t] == 0,
                                                    f"no_schedule_x_{h}_{i}_{j}_{t}")
                        elif t == schedule_len - 1:
                            # At the final scheduled time, only allow the scheduled arc.
                            if [i, j] == schedules[h][t]:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                f"end_x_{h}_{i}_{j}_{t}")
                            else:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 0,
                                                f"noend_x_{h}_{i}_{j}_{t}")
                        elif t > schedule_len - 1:
                            # Post-schedule: force a move from the last scheduled node to depot,
                            # or if at depot, remain there.
                            if i == schedules[h][schedule_len - 1][1] and j == 0:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                f"postschedule_x_{h}_{i}_{j}_{t}")
                            elif i == 0 and j == 0:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 1,
                                                f"stay_x_{h}_{i}_{j}_{t}")
                            else:
                                model.addConstr(machinery_flow.x[h, i, j, t] == 0,
                                                f"nopostschedule_x_{h}_{i}_{j}_{t}")
    model.update()


def from_schedule_to_hot_start_pruned(model, machinery_flow, schedules, Ih, T):
    """
    Provide a warm start (initial solution) based on a pre-computed schedule.
    For each machine h, the schedule provides the desired move [i,j] for each
    time period. In this version, we only set start values for those arcs
    that exist in the pruned decision variable dictionary.
    """
    for h in schedules.keys():
        schedule_len = len(schedules[h])
        for t in T:
            for i in Ih[h]:
                for j in Ih[h]:
                    # Proceed only if this arc exists in machinery_flow.x.
                    if (h, i, j, t) not in machinery_flow.x:
                        continue

                    if t > 0:
                        if t < schedule_len - 1:
                            if i == j:
                                if [str(i), str(j)] == schedules[h][t]:
                                    machinery_flow.idle[h, i, t].start = 1
                                elif [i, j] == schedules[h][t]:
                                    machinery_flow.working[h, i, t].start = 1
                            if i != j:
                                if [i, j] == schedules[h][t]:
                                    machinery_flow.x[h, i, j, t].start = 1
                                else:
                                    machinery_flow.x[h, i, j, t].start = 0
                        elif t == schedule_len - 1:
                            if [i, j] == schedules[h][t]:
                                machinery_flow.x[h, i, j, t].start = 1
                            else:
                                machinery_flow.x[h, i, j, t].start = 0
                        elif t > schedule_len - 1:
                            if i == schedules[h][schedule_len - 1][1] and j == 0:
                                machinery_flow.x[h, i, j, t].start = 1
                            elif i == 0 and j == 0:
                                machinery_flow.x[h, i, j, t].start = 1
                            else:
                                machinery_flow.x[h, i, j, t].start = 0
    model.update()


def from_coarse_schedule_to_initial_fine_hot_start(model, machinery_flow, coarse_schedules, Il, T_fine,
                                                   num_fine_steps_to_set: int):
    """
    Applies a warm start to the fine model using the initial movements from a coarse schedule.

    Only the first 'num_fine_steps_to_set' in the fine model are warm-started.
    For each fine time step t_f in this initial range, the t_f-th arc from the
    coarse schedule is used.

    Args:
        model: The Gurobi model object for the fine model.
        machinery_flow: The LoaderFlow object for the fine model.
        coarse_schedules: Dict mapping machine ID to its coarse schedule (list of [from, to] arcs).
                          This schedule is derived from the coarse model's solution.
        Il: Dictionary mapping machine ID to its list of accessible locations.
        T_fine: The range of time periods for the fine model.
        num_fine_steps_to_set: The number of initial fine time periods to warm-start.
    """
    print(f"Attempting partial warm-start for the first {num_fine_steps_to_set} fine steps.")

    for h in coarse_schedules.keys():
        coarse_schedule_for_h = coarse_schedules[h]
        coarse_schedule_len = len(coarse_schedule_for_h)

        # Iterate only for the first 'num_fine_steps_to_set' fine time periods
        for t_f in range(min(num_fine_steps_to_set, len(T_fine))):
            if t_f >= coarse_schedule_len:
                # Coarse schedule is shorter than num_fine_steps_to_set, stop for this machine.
                # print(f"  Machine {h}: Coarse schedule length ({coarse_schedule_len}) is less than t_f ({t_f}). Stopping warm-start for this t_f.")
                break

            # Get the t_f-th arc from the coarse schedule.
            # Note: Coarse schedule here is List[List[int]], e.g., [[0,0], [0,1], ...]
            # The distinction [str(i),str(j)] vs [i,j] for idle/working from heuristic schedules
            # is not present if coarse_schedules comes from extract_loader_schedule_list.
            scheduled_arc = coarse_schedule_for_h[t_f]
            i_scheduled = scheduled_arc[0]
            j_scheduled = scheduled_arc[1]

            # Set the scheduled arc's .start value if it exists in the fine model
            if (h, i_scheduled, j_scheduled, t_f) in machinery_flow.x:
                machinery_flow.x[h, i_scheduled, j_scheduled, t_f].start = 1
                # print(f"  Warm-start: x[{h},{i_scheduled},{j_scheduled},{t_f}] = 1")

                if i_scheduled == j_scheduled:  # It's a stay arc
                    # Default to setting 'working' for a stay arc from a coarse solution.
                    # The coarse solution's detailed idle/working state might be complex to pass,
                    # so warm-starting as 'working' is a reasonable default.
                    if (h, i_scheduled, t_f) in machinery_flow.working:
                        machinery_flow.working[h, i_scheduled, t_f].start = 1
                        # print(f"  Warm-start: working[{h},{i_scheduled},{t_f}] = 1")
                        if (h, i_scheduled, t_f) in machinery_flow.idle:
                            machinery_flow.idle[h, i_scheduled, t_f].start = 0  # Ensure idle is 0 if working
                    # else:
                    # print(f"  Warning: working[{h},{i_scheduled},{t_f}] not in machinery_flow.working for hotstart.")
            # else:
            # print(f"  Warning: Arc ({h},{i_scheduled},{j_scheduled},{t_f}) "
            #       f"from coarse schedule not in fine model's x variables. Cannot warm-start this arc.")

            # Set all other non-scheduled movements/stays at this t_f to 0 for clarity
            for i_iter in Il[h]:
                for j_iter in Il[h]:
                    if not (i_iter == i_scheduled and j_iter == j_scheduled):
                        if (h, i_iter, j_iter, t_f) in machinery_flow.x:
                            machinery_flow.x[h, i_iter, j_iter, t_f].start = 0
                            if i_iter == j_iter:  # Also set corresponding idle/working to 0
                                if (h, i_iter, t_f) in machinery_flow.idle:
                                    machinery_flow.idle[h, i_iter, t_f].start = 0
                                if (h, i_iter, t_f) in machinery_flow.working:
                                    machinery_flow.working[h, i_iter, t_f].start = 0
    model.update()


def from_coarse_schedule_to_selective_fine_hot_start(
        model: Model,
        machinery_flow: Any,  # Replace Any with your LoaderFlow class type
        coarse_schedules: Dict[Any, List[List[int]]],
        Il: Dict[Any, List[int]],
        T_fine: range,
        ratio: int,
        config: Dict[str, bool]
):
    """
    Applies a warm start to the fine model using specific depot-related movements
    from a coarse schedule. Correctly handles initial depot egress.

    Args:
        model: The Gurobi model object for the fine model.
        machinery_flow: The LoaderFlow object for the fine model.
        coarse_schedules: Dict mapping machine ID to its coarse schedule.
        Il: Dictionary mapping machine ID to its list of accessible locations.
        T_fine: The range of time periods for the fine model.
        ratio: Multiplier for time steps.
        config: Dict with boolean flags for setting different types of movements.
    """
    print(f"Attempting selective fine hot-start with ratio {ratio}. Config: {config}")
    if not T_fine or len(T_fine) == 0:  # Added check for empty T_fine
        print("Warning: T_fine is empty or invalid. Cannot apply selective hot-start.")
        return

    # print("\nWarmstart prints (Corrected)\n")
    # print("T_fine in warmstart function: ", T_fine)

    max_fine_t_val = T_fine[-1]

    for h in coarse_schedules.keys():
        coarse_schedule_for_h = coarse_schedules[h]
        if not coarse_schedule_for_h:
            continue

        # --- Helper to set fine steps for a given coarse step's arc ---
        def set_fine_steps_for_coarse_arc(t_c: int, i_coarse: int, j_coarse: int,
                                          is_initial_egress_move: bool = False) -> bool:
            """
            Sets .start for fine model variables.
            If is_initial_egress_move is True, for [0,j] from coarse:
              - Fine step 1: x[h,0,j,t_f1] = 1
              - Fine steps 2..ratio: x[h,j,j,t_f_subsequent] = 1 (stay at destination)
            Returns False if processing should stop.
            """
            for r_idx in range(ratio):
                t_f = t_c * ratio + r_idx
                if t_f > max_fine_t_val:
                    return False  # Stop if fine time exceeds model horizon

                # Determine the actual arc (i_fine, j_fine) for this fine step t_f
                i_fine, j_fine = i_coarse, j_coarse
                if is_initial_egress_move and r_idx > 0:
                    # For an egress move, after the first fine step, the machine stays at the destination
                    i_fine, j_fine = j_coarse, j_coarse  # Stay at destination j_coarse

                # Set the determined fine arc
                if (h, i_fine, j_fine, t_f) in machinery_flow.x:
                    print(f"Set X: ({h}, {i_fine}, {j_fine}, {t_f})")
                    machinery_flow.x[h, i_fine, j_fine, t_f].start = 1
                    if i_fine == j_fine:  # If it's a stay arc (either originally or modified for egress)
                        node_stay = i_fine
                        if (h, node_stay, t_f) in machinery_flow.working:
                            print(f"Set working: ({h}, {node_stay}, {t_f})")
                            machinery_flow.working[h, node_stay, t_f].start = 1
                        if (h, node_stay, t_f) in machinery_flow.idle:
                            machinery_flow.idle[h, node_stay, t_f].start = 0  # Assume working for simplicity
                # else:
                # print(f"  Warning: Fine arc ({h},{i_fine},{j_fine},{t_f}) not in machinery_flow.x")

                # Zero out other arcs for this (h, t_f)
                for i_iter in Il[h]:
                    for j_iter in Il[h]:
                        if not (i_iter == i_fine and j_iter == j_fine):  # Check against the arc we actually set
                            if (h, i_iter, j_iter, t_f) in machinery_flow.x:
                                machinery_flow.x[h, i_iter, j_iter, t_f].start = 0
                                if i_iter == j_iter:  # If it's a stay arc
                                    if (h, i_iter, t_f) in machinery_flow.idle:
                                        machinery_flow.idle[h, i_iter, t_f].start = 0
                                    if (h, i_iter, t_f) in machinery_flow.working:
                                        machinery_flow.working[h, i_iter, t_f].start = 0
            return True

        last_tc_initial_block = -1
        if config.get('set_initial_depot_stays', False):
            for t_c, arc in enumerate(coarse_schedule_for_h):
                if arc == [0, 0]:
                    # For [0,0] stays, is_initial_egress_move is False
                    if not set_fine_steps_for_coarse_arc(t_c, 0, 0, is_initial_egress_move=False): break
                    last_tc_initial_block = t_c
                else:
                    break

        t_c_initial_egress = last_tc_initial_block + 1
        initial_egress_processed = False
        if config.get('set_initial_depot_egress', False):
            if 0 <= t_c_initial_egress < len(coarse_schedule_for_h):
                arc_egress = coarse_schedule_for_h[t_c_initial_egress]
                if arc_egress[0] == 0 and arc_egress[1] != 0:  # Is a [0,j] move
                    # Pass is_initial_egress_move=True
                    if not set_fine_steps_for_coarse_arc(t_c_initial_egress, arc_egress[0], arc_egress[1],
                                                         is_initial_egress_move=True):
                        pass  # Error/stop condition already handled by set_fine_steps_for_coarse_arc
                    initial_egress_processed = True

        if config.get('set_terminal_depot_stays', False):
            for t_c_rev in range(len(coarse_schedule_for_h) - 1, -1, -1):
                if t_c_rev <= last_tc_initial_block: break
                if initial_egress_processed and t_c_rev == t_c_initial_egress: break

                arc = coarse_schedule_for_h[t_c_rev]
                if arc == [0, 0]:
                    # For [0,0] stays, is_initial_egress_move is False
                    if not set_fine_steps_for_coarse_arc(t_c_rev, 0, 0, is_initial_egress_move=False): break
                else:
                    break
    model.update()


def run_pl_heuristic_experiments(
    scenarios: List[Dict[str, Any]],
    instance_data: Dict[str, Any],
    sensitivity_scenarios: List[Dict[str, Any]],
    base_file_path: str,
    *,
    model_versions: Optional[List[Dict[str, Any]]] = None,
    vehicle_capacity_flag: bool = False,
    time_restriction: bool = False,
    verbose: bool = True,
    usage: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Execute the production‐loading heuristic over multiple scenarios, model versions,
    and sensitivity settings, collecting routes and schedules for benchmarking or warm starts.

    For each combination of:
      - a scenario (defined by number of fields, harvesters, loaders),
      - a model version (parameters like idle_time, restricted, access, working_hours, travel_time, t_p),
      - a sensitivity scenario (multipliers for cost, productivity, working hours),

    this function:
      1. Builds a unique key identifying the experiment.
      2. Extracts distance and travel‐time matrices, loader data, field attributes, and a precomputed
         production plan from `instance_data`.
      3. Computes scaled loader productivity (L_bar), clusters fields, and instantiates
         `PLBenchScheduler` to route each loader.
      4. For each loader, calls `create_production_plan_schedule_tau` to generate a daily‐aware,
         period‐by‐period loading schedule.
      5. Records the longest horizon T across loaders and stores routes, schedules, and the
         production plan under the experiment key.

    Args:
        scenarios: List of dictionaries, each specifying a scenario with keys
            'nr_fields', 'nr_h', and 'nr_l'.
        instance_data: Mapping from scenario_id to precomputed data dict, including
            'l_distance_matrices', 'l_tt_matrices', 'loader_data', field attributes,
            and 'production_plan'.
        sensitivity_scenarios: List of dicts, each with multipliers:
            'cost_multi', 'productivity_multi', 'working_hours_multi'.
        base_file_path: Base directory path for saving figures or intermediate output.
        model_versions: Optional list of dicts specifying model parameters:
            'idle_time', 'restricted', 'access', 'working_hours', 'travel_time', and 't_p'.
        vehicle_capacity_flag: If True, passes capacity enforcement to the schedule generator.
        time_restriction: If True, prevents scheduling beyond the provided daily limits.
        verbose: If True, prints progress and debug information to stdout.
        usage: If True, returns the aggregated results; otherwise returns None.

    Returns:
        A dict mapping each experiment key (str) to a sub-dict with:
            - 'loader_routes': Dict[int, List[int]] of each loader’s field route.
            - 'loader_schedule': Dict[int, List[List[int]]] of period‐by‐period moves.
            - 'T': int, the maximum schedule length across all loaders.
            - 'production_plan': pandas.DataFrame of daily goals used.
        Returned only when `usage=True`; otherwise returns None.
    """

    results = {}

    for scenario in scenarios:

        model_versions_results = {}
        max_delay_current_loop = 0

        # Extract a unique scenario identifier (e.g., using fields, harvesters, and loaders)

        scenario_id = f"{scenario['nr_fields']}_{scenario['nr_h']}_{scenario['nr_l']}"

        print("Scenario_ID: ", scenario_id)

        for model_version in model_versions:

            model_version_key = f"MV_{int(model_version['idle_time'])}_{int(model_version['restricted'])}_" \
                                f"{int(model_version['access'])}_{int(model_version['working_hours'])}_" \
                                f"{int(model_version['travel_time'])}_{model_version['t_p']}"

            for sens_scenario in sensitivity_scenarios:
                # Improved Sensitivity ID Naming
                sensitivity_id = f"S_{sens_scenario['cost_multi']}_{sens_scenario['productivity_multi']}_" \
                                 f"{sens_scenario['working_hours_multi']}"

                combined_key = f"{scenario_id}_{model_version_key}_{sensitivity_id}"

                prod_multi = sens_scenario["productivity_multi"]

                # Define Colors
                colors = ['blue', 'green', 'purple', 'grey', 'orange', 'red', 'teal', 'maroon', 'navy', 'olive']
                # Load Data
                t_p = model_version["t_p"]

                distance_matrices = instance_data[scenario_id]["l_distance_matrices"]
                tau = instance_data[scenario_id]["l_tt_matrices"]

                loader_data = instance_data[scenario_id]["loader_data"]

                field_locations = instance_data[scenario_id]["field_locations"]
                beet_yield = instance_data[scenario_id]["beet_yield"]
                field_size = instance_data[scenario_id]["field_size"]
                beet_volume = instance_data[scenario_id]["beet_volume"]

                production_plan = instance_data[scenario_id]["production_plan"]

                # Extract Data
                L_bar = {row['Maus Nr.']: row['MeanProductivityPerHour'] * t_p * prod_multi
                         for index, row in loader_data.iterrows()}

                if verbose:
                    print("L_bar: ", L_bar)

                nr_loaders = len(L_bar)
                nr_fields = loader_data.NumberOfFields.sum()
                n_clusters = max(math.floor(nr_fields / nr_loaders / 20), 1)

                if verbose:
                    print("N_clusters: ", n_clusters)

                # Instantiate the heuristic with scaled data
                heuristic = PLBenchScheduler(field_locations, beet_yield, field_size,
                                             loader_data, n_clusters, colors, base_file_path, verbose=False,
                                             save_figures=False)

                # Plan routes for all machines
                loader_routes = heuristic.route_fields_within_regions_and_visualize(plot=False)

                if verbose:
                    for loader_id, route in loader_routes.items():
                        print(f"Loader {loader_id} Route:", route)

                # Generate schedules for all harvesters and loaders
                loader_schedule = {}

                if verbose:
                    print("Production Plan:\n", production_plan)

                # Create Production Schedule
                for loader_id, route in loader_routes.items():
                    daily_limit = production_plan[f'{loader_id} Goal'].tolist()

                    loader_schedule[loader_id] = create_production_plan_schedule_tau(
                        route,
                        beet_volume,
                        L_bar[loader_id],
                        daily_limit,
                        t_p, tau,
                        machine_id=loader_id,
                        vehicle_capacity_flag=vehicle_capacity_flag,
                        time_restriction=time_restriction)

                if verbose:
                    print("PL Bench Schedule")
                    for loader_id, schedule in loader_schedule.items():
                        print(f"Loader {loader_id} Schedule:", schedule)

                T = max(len(schedule) for schedule in loader_schedule.values())

                # Create a results dictionary for this scenario
                results[combined_key] = {
                    "loader_routes": loader_routes,
                    "loader_schedule": loader_schedule,
                    "T": T,
                    "production_plan": production_plan
                }

                # results[combined_key] = model_versions_results

    if usage:
        return results


def extract_loader_schedule_list(decision_variables):
    """
    Extracts the loader schedule from Gurobi decision variables into a
    nested list format, ordered by time.

    Args:
        decision_variables (dict): Dictionary containing the solution values
                                   for variables, specifically requiring 'loader_flow_x'.
                                   Example: {'loader_flow_x': {(l,i,j,t): val, ...}}

    Returns:
        dict: A dictionary where keys are loader IDs (int or str as stored in keys)
              and values are lists of [from_location, to_location] pairs,
              ordered chronologically by time t.
              Example: {51: [[0, 0], [0, 15], [15, 15], ...], 73: [...]}
              Returns an empty dictionary if 'loader_flow_x' is not found.
    """
    if 'loader_flow_x' not in decision_variables:
        print("Warning: 'loader_flow_x' key not found in decision_variables.")
        return {}

    loader_x_vars = decision_variables['loader_flow_x']

    # 1. Identify all unique loaders and time periods from the solution
    loaders = sorted(list(set(l for l, i, j, t in loader_x_vars.keys())))
    time_periods = sorted(list(set(t for l, i, j, t in loader_x_vars.keys())))

    if not loaders or not time_periods:
        print("Warning: No loaders or time periods found in 'loader_flow_x' data.")
        return {}

    # 2. Initialize the schedule dictionary with empty lists
    schedule = {l: [] for l in loaders}

    # 3. Iterate through time periods chronologically
    for t in time_periods:
        active_arcs_found_for_t = set()  # Track loaders processed for this time t
        for l in loaders:
            found_arc_for_loader_t = False
            # Search for the active arc for loader l at time t
            # Iterate through relevant subset of keys for efficiency if needed,
            # but direct check is usually fine.
            for (l_key, i_key, j_key, t_key), value in loader_x_vars.items():
                if l_key == l and t_key == t and value > 0.5:
                    schedule[l].append([i_key, j_key])
                    active_arcs_found_for_t.add(l)
                    found_arc_for_loader_t = True
                    break  # Found the active arc for this (l, t), move to next loader

            # Optional: Check if an arc was found for every loader at this time step
            # This might indicate an issue if a loader has no assigned arc,
            # although the model constraint should prevent this for feasible solutions.
            # if not found_arc_for_loader_t:
            #    print(f"Warning: No active arc found for loader {l} at time {t}.")
            #    # Decide how to handle: append None, previous state, error?
            #    # schedule[l].append(None) # Example: append None

    # Final check: Ensure all loaders have schedules of the same length (equal to num time periods)
    num_time_periods = len(time_periods)
    for l, moves in schedule.items():
        if len(moves) != num_time_periods:
            print(
                f"Warning: Loader {l} schedule length ({len(moves)}) does not match number of time periods ({num_time_periods}).")

    return schedule


# --- Example Usage ---

# 1. Load your results data (as shown in your example code)
# Assuming 'results' is loaded from your pickle file
# Example:
# solution_file_path = '../data/results/reporting/results_f_double_simulated_h.pkl'
# with open(solution_file_path, 'rb') as f:
#     results = pickle.load(f)

# key = '15_2_2_MV_1_1_0_0_1_2_S_1_1_1' # Use the correct key for your desired result

# if key in results:
#     decision_vars_actual = results[key]['decision_variables']

#     # 2. Extract the schedule
#     loader_schedule = extract_loader_schedule_list(decision_vars_actual)

#     # 3. Print the result
#     print("\nExtracted Loader Schedule (List Format):")
#     if loader_schedule:
#         for loader_id, moves in sorted(loader_schedule.items()):
#             print(f"Loader {loader_id}: {moves}")
#     else:
#         print("No schedule generated.")

# else:
#     print(f"Key '{key}' not found in loaded results.")


# --- Example with dummy data ---
"""
decision_variables_example = {
    'loader_flow_x': {
        (51, 0, 0, 0): 1.0, (73, 0, 0, 0): 1.0,
        (51, 0, 15, 1): 1.0, (73, 0, 10, 1): 0.0,  # Made one zero for testing
        (51, 15, 15, 2): 1.0, (73, 0, 10, 2): 1.0,  # Added the active arc for 73, t=1
        (51, 15, 15, 3): 1.0, (73, 10, 12, 3): 1.0,
        (51, 15, 8, 4): 1.0, (73, 12, 12, 4): 1.0,
        (73, 10, 10, 2): 1.0,  # Added missing 73, t=2
    },
    'beet_flow_b': {(0, 0, 0, 0): 100},  # Dummy entry to show other variables exist
    'idle': {}, 'working': {}
}

print("\n--- Example with Dummy Data ---")
dummy_schedule = extract_loader_schedule_list(decision_variables_example)
if dummy_schedule:
    for loader_id, moves in sorted(dummy_schedule.items()):
        print(f"Loader {loader_id}: {moves}")
else:
    print("No schedule generated.")

print("\n--- Example with Actual Data ---\n")
dummy_schedule = extract_loader_schedule_list(decision_variables)
if dummy_schedule:
    for loader_id, moves in sorted(dummy_schedule.items()):
        print(f"Loader {loader_id}: {moves}\n")
else:
    print("No schedule generated.")
"""


def build_valid_arc_sets_per_fine_step(
        coarse_plan_list: List[List[int]],  # INPUT: Accept List of Lists
        ratio: int = 2,
        buffer_pct: float = 0.1
) -> Dict[int, Set[Arc]]:  # OUTPUT: Still returns Set of Tuples (Arc)
    """
    Transforms a single loader's coarse schedule into a dictionary describing,
    for every fine-grained time index, which arcs (as tuples) remain valid.

    Includes original arcs within the buffer window AND corresponding stay arcs
    for any allowed travel arcs.

    Args:
        coarse_plan_list: Sequence of [from_loc, to_loc] lists, one per coarse slot.
        ratio: Number of fine slots per coarse slot.
        buffer_pct: Relative half-width [0, 1] of the symmetric time window.

    Returns:
        A dictionary mapping fine time index (t_f) to the set of valid arcs (tuples).
    """
    if not coarse_plan_list:
        return {}

    try:
        coarse_plan_tuples: List[Arc] = [tuple(arc) for arc in coarse_plan_list if
                                         len(arc) == 2]  # Ensure valid pairs before tuple conversion
        if len(coarse_plan_tuples) != len(coarse_plan_list):
            print(f"Warning: Some elements in coarse_plan_list were not valid pairs. Input: {coarse_plan_list}")
            if not coarse_plan_tuples: return {}  # Return empty if no valid pairs found
    except (TypeError, ValueError) as e:
        print(f"Error converting coarse plan lists to tuples: {e}. Input: {coarse_plan_list}")
        return {}

    Hc = len(coarse_plan_tuples)
    if Hc == 0:
        return {}

    Hf = Hc * ratio
    buf = max(0, min(Hc, round(buffer_pct * Hc)))
    unique_arcs: Set[Arc] = set(coarse_plan_tuples)
    allowed_coarse_indices: Dict[Arc, Set[int]] = defaultdict(set)

    # Determine allowed coarse indices for original arcs based on buffer
    if buffer_pct >= 1.0:
        all_coarse_t = set(range(Hc))
        for arc in unique_arcs:
            allowed_coarse_indices[arc] = all_coarse_t
    else:
        for k_c, arc in enumerate(coarse_plan_tuples):
            lo = max(0, k_c - buf)
            hi = min(Hc - 1, k_c + buf)
            allowed_coarse_indices[arc].update(range(lo, hi + 1))

    # Project onto the finer grid AND add corresponding stay arcs

    valid_arcs_at_fine_t: Dict[int, Set[Arc]] = {t_f: set() for t_f in range(Hf)}

    for t_f in range(Hf):
        k_c = t_f // ratio
        # Find all original arcs allowed at this coarse index k_c
        for original_arc in unique_arcs:
            if k_c in allowed_coarse_indices[original_arc]:
                # 1. Add the original allowed arc itself
                valid_arcs_at_fine_t[t_f].add(original_arc)

                # 2. If it's a travel arc (i != j), add stay arcs (i,i) and (j,j)
                i, j = original_arc
                if i != j:
                    # Add stay arc at the origin location
                    valid_arcs_at_fine_t[t_f].add((i, i))
                    # Add stay arc at the destination location
                    valid_arcs_at_fine_t[t_f].add((j, j))

    print(f"USED PCT: {buffer_pct} buffer to prune variables.")
    return valid_arcs_at_fine_t


def generate_valid_arcs_for_all_loaders(
        decision_variables: Dict[str, Any],
        ratio: int,
        buffer_pct: float
) -> Dict[int, Dict[int, Set[Arc]]]:
    """
    Generates the time-dependent valid arc sets for ALL loaders based on a
    coarse schedule extracted from decision variables.

    This function connects `extract_loader_schedule_list` and
    `build_valid_arc_sets_per_fine_step`.

    Args:
        decision_variables: The dictionary containing Gurobi solution variables,
                            specifically requiring 'loader_flow_x'.
        ratio: Number of fine time steps per coarse time step (e.g., 2 for 2h->1h).
        buffer_pct: The buffer percentage (0.0 to 1.0) to use when determining
                    arc validity in the fine time steps.

    Returns:
        A dictionary where keys are loader IDs. Each value is another dictionary
        mapping a fine time step index (int) to a set of valid arcs (Set[Arc])
        for that loader at that fine time.
        Example: {
            51: {0: {(0,0), (0,15)}, 1: {(0,15), (15,15)}, ...},
            73: {0: {(0,0)}, 1: {(0,10)}, ...}
        }
        Returns an empty dictionary if the initial schedule extraction fails.
    """
    print("Step A: Extracting coarse schedules for all loaders...")
    # 1. Extract schedule per loader: Dict[LoaderID, List[List[int]]]
    coarse_schedules_per_loader = extract_loader_schedule_list(decision_variables)

    print(f"  Extracted Schedule: \n {coarse_schedules_per_loader}.")

    if not coarse_schedules_per_loader:
        print("Extraction failed or no schedules found. Returning empty valid arc set.")
        return {}

    print(f"Step B: Generating valid fine arcs for {len(coarse_schedules_per_loader)} loaders...")
    # 2. Initialize the final result dictionary
    valid_arcs_all_loaders: Dict[int, Dict[int, Set[Arc]]] = {}

    # 3. Process each loader's schedule
    for loader_id, schedule_list in coarse_schedules_per_loader.items():
        if not schedule_list:
            print(f"  Skipping loader {loader_id} due to empty coarse schedule.")
            continue  # Skip if a loader somehow has an empty schedule list

        # Apply the second function to this loader's schedule
        # Result: Dict[FineTime, Set[Arc]] for this specific loader_id
        valid_arcs_for_loader = build_valid_arc_sets_per_fine_step(
            coarse_plan_list=schedule_list,
            ratio=ratio,
            buffer_pct=buffer_pct
        )

        # Store the result in the main dictionary under the loader_id
        valid_arcs_all_loaders[loader_id] = valid_arcs_for_loader
        # Optional: Add a small progress indicator if many loaders
        # print(f"  Valid Combinations Loader{loader_id}: \n {valid_arcs_all_loaders[loader_id]}.\n")
        # print(f"  Processed loader {loader_id}.")

    # print("Step C: Finished generating valid fine arcs for all loaders.")
    # 4. Return the final combined dictionary
    return valid_arcs_all_loaders


def get_scenario_id(size_scn: Dict) -> str:
    """Generates the scenario ID string used to index instance_data."""
    # Example implementation (ADAPT TO YOUR instance_data keys)
    return f"{size_scn.get('nr_fields', 'X')}_{size_scn.get('nr_h', 'X')}_{size_scn.get('nr_l', 'X')}"


def make_key(size_scn: Dict, model_ver: Dict, sens_scn: Dict) -> str:
    """
    Generates a unique key based on scenario parameters, following the specified format.

    Args:
        size_scn: Dictionary with size parameters (nr_fields, nr_h, nr_l).
        model_ver: Dictionary with model version parameters (idle_time, restricted, etc., t_p).
        sens_scn: Dictionary with sensitivity parameters (cost_multi, productivity_multi, etc.).

    Returns:
        A unique string identifier for the combination.
    """
    # 1. Create scenario_id part
    scenario_id = f"{size_scn.get('nr_fields', 'X')}_{size_scn.get('nr_h', 'X')}_{size_scn.get('nr_l', 'X')}"

    # 2. Create model_version_key part
    model_version_key = (
        f"MV_{int(model_ver.get('idle_time', 0))}_"
        f"{int(model_ver.get('restricted', 0))}_"
        f"{int(model_ver.get('access', 0))}_"
        f"{int(model_ver.get('working_hours', 0))}_"
        f"{int(model_ver.get('travel_time', 0))}_"
        f"{model_ver.get('t_p', 'X')}"
    )

    # 3. Create sensitivity_id part
    sensitivity_id = (
        f"S_{sens_scn.get('cost_multi', 1)}_"
        f"{sens_scn.get('productivity_multi', 1)}_"
        f"{sens_scn.get('working_hours_multi', 1)}"
    )

    # 4. Combine the parts
    combined_key = f"{scenario_id}_{model_version_key}_{sensitivity_id}"
    return combined_key


def prepare_loader_flow_data(
        instance_raw: Dict[str, Any],
        model_ver: Dict[str, Any],
        *,
        sens: Dict[str, float] = None,
        T_heuristic: Optional[int] = None,
        loader_routes_heuristic: Optional[Dict[int, List[int]]] = None,
        valid_arcs_at_time: Optional[Dict[int, Dict[int, Set[Tuple[int, int]]]]] = None,
        inventory_levels: float = 0,
        last_period_beet_volume: dict = None

) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepares data for solve_loader_flow, incorporating optional heuristic results.

    Args:
        instance_raw: Dictionary with base instance data.
        sens: Dictionary with sensitivity parameters.
        model_ver: Dictionary with model version details (e.g., t_p).
        T_heuristic: Optional time horizon length determined by a heuristic.
                     If None, T_horizon must be derivable from instance_raw or default.
        loader_routes_heuristic: Optional dictionary mapping loader ID to a
                                 predefined route (list of nodes) from a heuristic.
                                 Takes precedence over valid_arcs_at_time if both provided,
                                 depending on LoaderFlow logic (Check LoaderFlow).
        # --- Docstring ---
        valid_arcs_at_time: Optional dictionary mapping loader ID to another
                            dictionary, which maps fine time index (t) to a
                            set of valid (from_loc, to_loc) arcs allowed for
                            that loader at that specific time. Used for pruning.
                            Example: {51: {0: {(0,0), (0,15)}, 1: ...}, 73: ...}
        # Previous was: valid_combinations: Optional Dictionary setting valid combinations per time period
        inventory_levels:   Sets initial inventory levels for beets in storage
        last_period_beets:  Dict, holding a beet on field volume goal at last period -> (i,0,0,max(T))

    Returns:
        tuple[Dict[str, Any], Dict[str, Any]]:
            - instance_dict: The original raw instance data.
            - derived_dict: Derived parameters and pre-packaged constructor args,
                            incorporating heuristic results if provided.
    """

    # ------------- unpack raw --------------------------------------------------------
    I, L, Il = instance_raw["I"], instance_raw["L"], instance_raw["Il"]
    t_p = model_ver["t_p"]
    tau = instance_raw['l_tt_matrices']  # Assuming this is the correct key

    # ------------- Determine Time Horizon (T_horizon) --------------------------------
    if T_heuristic is not None:
        T_horizon = T_heuristic
        print(f"Using heuristic time horizon: T = {T_horizon}")
    else:
        # Fallback: Try to get from instance_raw or set a default/raise error
        # Example fallback (adjust as needed):
        T_horizon = instance_raw.get("T_horizon_default", 74)
        if T_horizon is None:
            # If T is critical and not provided by heuristic, raise error
            raise ValueError("Time horizon T must be provided via T_heuristic or instance_raw['T_horizon_default']")
        print(f"Warning: Using default/instance time horizon: T = {T_horizon}")
        # T_horizon = 2 # Original hardcoded value - likely needs better handling

    T = range(T_horizon)  # Define the time range based on the determined horizon

    # ------------- derived productivity / costs -------------------------------------
    loader_data = instance_raw["loader_data"]
    # Ensure 'Maus Nr.' and 'MeanProductivityPerHour' exist in loader_data columns
    if not all(col in loader_data.columns for col in ['Maus Nr.', 'MeanProductivityPerHour']):
        raise ValueError("Missing required columns 'Maus Nr.' or 'MeanProductivityPerHour' in loader_data")

    if sens:
        L_bar = {r["Maus Nr."]: r["MeanProductivityPerHour"] * t_p *
                                sens["productivity_multi"] for _, r in loader_data.iterrows()}
    else:
        L_bar = {r["Maus Nr."]: r["MeanProductivityPerHour"] * t_p for _, r in loader_data.iterrows()}

    # Ensure 'production_volume_per_day' exists
    if "production_volume_per_day" not in instance_raw:
        raise ValueError("Missing 'production_volume_per_day' in instance_raw")
    production_volume_per_day = instance_raw["production_volume_per_day"]

    Pmin_base = (production_volume_per_day / 24) * t_p

    # Use the determined T_horizon for production bounds
    P_bar_min = [0 if t < 2 else Pmin_base for t in T]
    P_bar_max = [0 if t < 1 else 1.01 * Pmin_base for t in T]

    # Ensure 'c_l' exists
    if "c_l" not in instance_raw:
        raise ValueError("Missing 'c_l' in instance_raw")

    c_l = instance_raw["c_l"].copy()  # Assuming c_l needs copying and potential modification

    if sens:
        for key, item in c_l.items():
            c_l[key] = c_l[key] * sens["cost_multi"]

    BEET_PRICE = 1.5
    # Define penalty rate based on beet price
    penalty_rate = BEET_PRICE * 4

    # Get production plan
    production_plan_full = instance_raw["production_plan"]

    # Vehicle capacity
    vehicle_cap = {}
    for l in L:
        # Construct the list of columns to select
        loader_goal_columns = [f"{l} Goal"]

        # Check which of these columns exist in the production_plan_full
        existing_loader_goal_columns = [col for col in loader_goal_columns if col in production_plan_full.columns]

        # Subset the production_plan
        # print("Production Plan in prepare_loader_flow:", production_plan_full)
        try:
            production_plan = production_plan_full[['Date', 'Holiday'] + existing_loader_goal_columns].copy()
        except KeyError:
            print("Warning: No holiday column in production plan")
            production_plan = production_plan_full[['Date'] + existing_loader_goal_columns].copy()

        # Sum the loading goals for the first seven days
        capacity_sum = production_plan.loc[
                       0:6, existing_loader_goal_columns].sum().iloc[0] if not production_plan.loc[
                                                                               0:6,
                                                                               existing_loader_goal_columns].empty else 0
        vehicle_cap[l] = float(capacity_sum)  # Retrieve scalar and convert to float if necessary

    # inventory goals
    inventory_goal = instance_raw.get("inventory_goal", None)


    # Holidays:
    if "holidays" not in instance_raw:
        print("Use custom holiday calculation in prepare_loader_flow_data")
        holidays = production_plan_holiday_to_t(production_plan_full, periods_per_day=int(24 / t_p))
    else:
        holidays = instance_raw["holidays"]

    # ------------- Prepare constructor arguments ------------------------------------

    # Arguments for BeetFlow constructor
    beet_flow_args = dict(
        I=I,
        T=T,
        beet_volume=instance_raw["beet_volume"],
        inventory=inventory_levels,
        last_period_beet_volume=last_period_beet_volume
    )

    # Arguments for LoaderFlow constructor
    loader_flow_args = dict(
        L=L,
        Il=Il,
        I=I,
        c_l=c_l,  # Pass the potentially modified cost dict
        T=T,  # Use the determined T
        tau=tau,
        access=instance_raw.get("loader_access", None),  # Get access info if available
        working_hours=instance_raw.get("loader_working_hours", None),  # Get working hours if available
        holidays=holidays,
        v_type=model_ver.get("v_type", "binary"),
        time_period_length=t_p,
        # --- Integrate Heuristic Routes ---
        # Pass the heuristic routes if they were provided to this function
        loader_routes=loader_routes_heuristic,
        valid_arcs_at_time=valid_arcs_at_time,
        beet_volume=instance_raw["beet_volume"],  # Source beet_volume
        loader_rates=L_bar
    )
    # Optional: Add a print statement if routes are being used
    if loader_routes_heuristic:
        print(f"Passing {len(loader_routes_heuristic)} heuristic loader routes to LoaderFlow.")

    # ------------- Assemble derived dictionary --------------------------------------
    derived = dict(
        # ----- model-wide scalars / vectors -----------------------------------------
        T=T,  # Store the final range T
        L_bar=L_bar,
        P_bar_min=P_bar_min,
        P_bar_max=P_bar_max,
        IC=Pmin_base * (24 / t_p * 4) if t_p > 0 else 0,  # Calculate IC based on Pmin_base and t_p (e.g., 4 days)
        c_l=c_l,  # Store the cost dict used
        c_s=0.01 * t_p,  # Storage cost per period
        buffer_pct=0.075,   # Set pruning buffer param
        BEET_PRICE=BEET_PRICE,
        penalty_rate=penalty_rate,
        LAMBDA=sens.get("LAMBDA", 0.75),  # Get LAMBDA from sens or default
        t_p=t_p,
        vehicle_capacity=vehicle_cap,
        production_volume_per_day=production_volume_per_day,
        inventory_goal=inventory_goal,
        # ----- constructor kwargs ---------------------------------------------------
        beet_flow_args=beet_flow_args,
        loader_flow_args=loader_flow_args,  # Contains loader_routes if provided
    )
    # Return the original raw instance and the newly assembled derived dictionary
    return instance_raw, derived


def solve_loader_flow(
        instance: Dict[str, Any],
        derived: Dict[str, Any],
        name: str,
        *,
        hotstart_solution: Dict[str, Any] = None,
        enforce_solution: Dict[str, Any] = None,
        # New parameters for time aggregation partial warmstart
        coarse_hotstart_schedule: Optional[Dict[str, List[List[int]]]] = None,
        coarse_hotstart_config: Optional[Dict[str, Any]] = None,
        # End new
        FIXED_SOLVER_PARAMS: Dict[str, Any] = None,
        verbose: bool | None = None,
        combined_key: str | None = None,
        base_file_path: str = "../../data/",
        v_type: str = "binary",
        vehicle_capacity_flag: bool = False,
        restricted_flow_flag: bool = True,
        last_period_restricted_flag: bool = False,
        add_min_beet_restriction_flag: bool = False,
        inventory_flag: bool = False,
        inventory_cap_flag: bool = True,
        holidays_flag: bool = False
) -> Dict[str, Any]:
    """
    Solves the Multi-Commodity Flow Harvest Planning Model using Gurobi.

    This function sets up and optimizes a Gurobi model based on the provided instance
    and derived data, including various configurable features like restricted flow,
    inventory management, and holiday constraints. It also supports warm-starting
    the solution with pre-computed schedules (either fine-grained or coarse-grained
    for time aggregation).

    Args:
        instance (Dict[str, Any]): A dictionary containing raw instance data such as
            field locations (I), loaders (L), accessible locations per loader (Il),
            beet volume at fields (beet_volume), and travel time matrices (tau).
        derived (Dict[str, Any]): A dictionary containing derived parameters and
            pre-processed data for the model, including time horizon (T),
            loader productivity (L_bar), production bounds (P_bar_min, P_bar_max),
            inventory capacity (IC), various costs (c_l, c_s), beet price, penalty rates,
            and time period length (t_p). It also contains arguments for initializing
            BeetFlow and LoaderFlow objects.
        name (str): The name for the Gurobi model and for saving results.
        hotstart_solution (Dict[str, Any], optional): A dictionary containing a
            fine-grained pre-computed schedule (e.g., 'loader_schedule') to be used
            as a warm start for the solver. Defaults to None.
        enforce_solution (Dict[str, Any], optional): A dictionary containing a
            fine-grained pre-computed schedule (e.g., 'loader_schedule') to be
            enforced as hard constraints in the model. Defaults to None.
        coarse_hotstart_schedule (Optional[Dict[str, List[List[int]]]]): A dictionary
            mapping loader IDs to a list of coarse-grained [from_loc, to_loc] movements.
            Used for partial warm-starting in time aggregation scenarios. Defaults to None.
        coarse_hotstart_config (Optional[Dict[str, Any]]): Configuration for the
            coarse warm-start, including 'type' (e.g., 'selective_depot_related')
            and other parameters like 'ratio', 'set_initial_depot_stays', etc.
            Defaults to None.
        FIXED_SOLVER_PARAMS (Dict[str, Any], optional): A dictionary of Gurobi solver
            parameters to be set (e.g., 'MIPGap', 'TimeLimit'). Defaults to None.
        verbose (bool | None, optional): If True, enables verbose output during execution.
            Defaults to None, letting Gurobi's own verbosity settings apply or be overridden
            by `FIXED_SOLVER_PARAMS`.
        combined_key (str | None, optional): A unique identifier for the current scenario
            combination, used for indexing results or hotstart/enforce solutions. Defaults to None.
        base_file_path (str): The base directory path for saving results.
            Defaults to "../../data/".
        v_type (str): Variable type for Gurobi variables, e.g., "binary", "continuous",
            or "integer". Defaults to "binary".
        vehicle_capacity_flag (bool): If True, activates vehicle capacity constraints.
            Defaults to False.
        restricted_flow_flag (bool): If True, activates hard restricted flow constraints
            (i.e., a field must be finished before moving to another). Defaults to True.
        last_period_restricted_flag (bool): If True, activates a modified restricted
            flow constraint focusing only on the last period. Mutually exclusive with
            `restricted_flow_flag`. Defaults to False.
        add_min_beet_restriction_flag (bool): If True, adds minimum beet quantity
            restriction constraints. Defaults to False.
        inventory_flag (bool): If True, activates inventory goal constraints.
            Requires `vehicle_capacity_flag` to be True. Defaults to False.
        inventory_cap_flag (bool): If True, adds an upper limit to the inventory goal.
            Requires `inventory_flag` to be True. Defaults to True.
        holidays_flag (bool): If True, activates holiday constraints. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the results of the optimization, including:
            - 'status' (str): The Gurobi optimization status (e.g., 'optimal', 'infeasible').
            - 'parameters' (Dict[str, Any]): A dictionary of key model parameters used.
            - 'decision_variables' (Dict[str, Dict[Any, float]]): A dictionary of solved
              decision variables and their values (e.g., 'loader_flow_x', 'beet_flow_b', 'unmet_demand').
              Only non-zero 'x', 'working', 'idle', 'y_in', 'y_out' variables are stored for memory efficiency.
              'started', 'finished', and 'restricted' variables are included if activated.
    """


    #
    # Load Data
    #

    I, L, Il = instance["I"], instance["L"], instance["Il"]
    beet_volume = instance["beet_volume"]
    tau = instance["l_tt_matrices"]
    T = derived["T"]
    L_bar = derived["L_bar"]
    P_bar_min = derived["P_bar_min"]
    P_bar_max = derived["P_bar_max"]
    IC = derived["IC"]
    c_l = derived["c_l"]
    c_s = derived["c_s"]
    BEET_PRICE = derived["BEET_PRICE"]
    penalty_rate = derived["penalty_rate"]
    t_p = derived["t_p"]
    LAMBDA = derived["LAMBDA"]
    holidays = derived["loader_flow_args"]["holidays"]
    epsilon = 1e-7

    # Optional data
    if vehicle_capacity_flag:
        vehicle_cap = derived["vehicle_capacity"]
        print("vehicle_cap: ", vehicle_cap)

        if inventory_flag:
            print("Vehicle_capacity_flag and inventory_flag activated")
            # load data
            production_volume_per_day = derived["production_volume_per_day"]
            initial_inventory = derived["beet_flow_args"]["inventory"]
            inventory_goal_derived = derived["inventory_goal"]

            if not inventory_goal_derived:
                print("Use Calculated Inventory Goal")
                # Calculate inventory goals based on vehicle loading goals + inventory levels minus demand for one workweek
                inventory_goals = sum(vehicle_cap.values()) + initial_inventory - (production_volume_per_day * 6)
                # print("WARING UPDATE INVENTORY GOAL THIS IS AN EXPERIMENT")
                # production_volume_per_day / 10
            else:
                print("Use derived inventory goal")
                inventory_goals = inventory_goal_derived

            print("Inventory goal in solver before correction: ", inventory_goals)

            if inventory_goals < 0:
                print("\nWarning: inventory_goals below 0! Now, adapted to production_volume_per_day.\n")
                inventory_goals = production_volume_per_day

            print("production_volume_per_day: ", production_volume_per_day)
            print("vehicle capcity in sovler: ", vehicle_cap)
            print("initial inventory", initial_inventory)
            print("inventory goal in solver after correction: ", inventory_goals)

    else:
        print("\nWarning: Vehicle_capacity and inventory goal NOT activated\n")

    # --- Gurobi model -----------------------------------------------------------------
    m = Model(name)

    # Initialize
    beet_flow = BeetFlow(m, **derived["beet_flow_args"])
    loader_flow = LoaderFlow(m, **derived["loader_flow_args"])

    unmet_demand = m.addVars(T, vtype=GRB.CONTINUOUS, lb=0, ub=13500, name="unmet_demand")

    # --- Core constraints (all features on) -------------------------------------------
    # Beetflow
    beet_flow.add_load_only_flow_conservation_constraint()

    if add_min_beet_restriction_flag:
        if restricted_flow_flag or last_period_restricted_flag:
            print("WARNING: Restriction Activated add_min_beet_restriction in last period will have limited effects."
                  "Try implementing without restrictions to minimise scale of model")
        beet_flow.add_min_beet_restriction()

    # Loaderflow (core, restricted, idle, partial travel and holiday)
    loader_flow.add_core_constraints()

    #
    # Restricted
    #
    if last_period_restricted_flag:
        # Check error
        if restricted_flow_flag:
            raise ValueError("last_period_restricted_flag and restricted_flow_flag are active in solve_loader_flow()."
                             "Only activate at most one.")
        print("\nWarning: last_period_restricted_flag activated, non-standard!\n")

        # This call now creates started/finished/restricted only for valid (l,i,t)
        loader_flow.add_last_period_restricted()

        # Additional Restricted Constraints (Adapted)

        # Define Started: Link started variable to incoming flow from non-depot locations
        # Iterate only over the keys for which the 'started' variable exists
        print("Adding 'Started' indicator constraints...")
        if loader_flow.started:  # Check if the dictionary is not empty
            for (l, j, t) in loader_flow.started.keys():
                # The key (l, j, t) guarantees j != 0 because 'started' is based on valid_stay_lit_no_depot

                # Build the list of valid incoming arcs to j from non-depot i, up to time t'.
                # This part correctly checks against the pruned self.x
                valid_arcs_cumulative = [
                    (l, i, j, t_prime)
                    # Iterate over potential origins 'i' for this loader 'l'
                    for i in loader_flow.Il[l]
                    # Iterate over all relevant time points up to and including 't'
                    for t_prime in T
                    # Assuming loader_flow.T is max time, and 0 is start. Adjust range if needed.
                    if t_prime <= t
                    # Ensure origin is not depot and the arc exists in the pruned x variables
                    if (l, i, j, t_prime) in loader_flow.x  # i != 0 and
                ]

                # Sum over those existing arcs.
                # Only sum if valid_arcs_cumulative is not empty, otherwise quicksum on empty list is 0, which is fine.
                cumulative_flow = quicksum(loader_flow.x[arc] for arc in valid_arcs_cumulative)

                # Add indicator constraints - now referencing an existing started[l, j, t]
                m.addGenConstrIndicator(
                    loader_flow.started[l, j, t], True,
                    cumulative_flow, GRB.GREATER_EQUAL, 1.0 - epsilon,  # Use tolerance
                    name=f"ind_start_1_{l}_{j}_{t}"
                )

                m.addGenConstrIndicator(
                    loader_flow.started[l, j, t], False,
                    cumulative_flow, GRB.LESS_EQUAL, 0.0 + epsilon,  # Use tolerance
                    name=f"ind_start_0_{l}_{j}_{t}"
                )
        print("Adding 'Finished' definition constraints...")

        if loader_flow.finished:  # Check if the dictionary is not empty
            for (l, i, t) in loader_flow.finished.keys():
                # The key (l, i, t) guarantees i != 0 here

                # Ensure the corresponding beet flow variable exists (should generally exist)
                # Note: BeetFlow 'b' indices are (field, source, sink, time)
                beet_var_key = (i, 0, 0, t)  # Volume remaining at field i (source=0) at heap (sink=0) at time t
                if beet_var_key not in beet_flow.b:
                    print(
                        f"Warning: Beet variable {beet_var_key} not found for Finished constraint ({l},{i},{t}). "
                        f"Skipping.")
                    continue

                # Add constraints - now referencing an existing finished[l, i, t]
                # If finished=1, beet_volume must be near 0.
                # If finished=0, beet_volume can be > 0.
                m.addConstr(
                    beet_flow.b[i, 0, 0, t] <= (beet_volume[i] + epsilon) * (
                            1 - loader_flow.finished[l, i, t]),
                    name=f"finish_upper_{l}_{i}_{t}"
                )
                # This lower bound might be too strict if epsilon is small and flow is continuous?
                # Maybe remove if causing issues, or adjust epsilon.
                # It tries to force finished=0 if *any* volume > epsilon remains.
                m.addConstr(
                    beet_flow.b[i, 0, 0, t] >= epsilon * (1 - loader_flow.finished[l, i, t]),
                    name=f"finish_lower_{l}_{i}_{t}"
                )
        else:
            print("Skipping 'Finished' definition constraints as no 'finished' variables exist.")

    if restricted_flow_flag:
        # This call now creates started/finished/restricted only for valid (l,i,t)
        loader_flow.add_hard_restricted()

        # Additional Restricted Constraints (Adapted)

        # Define Started: Link started variable to incoming flow from non-depot locations
        # Iterate only over the keys for which the 'started' variable exists
        print("Adding 'Started' indicator constraints...")
        if loader_flow.started:  # Check if the dictionary is not empty
            for (l, j, t) in loader_flow.started.keys():
                # The key (l, j, t) guarantees j != 0 because 'started' is based on valid_stay_lit_no_depot

                # Build the list of valid incoming arcs to j from non-depot i, up to time t.
                # This part correctly checks against the pruned self.x
                valid_arcs = [
                    (l, i, j, t_bar)
                    for t_bar in range(t + 1)
                    # Iterate over potential origins 'i' for this loader 'l'
                    for i in loader_flow.Il[l]
                    # Ensure origin is not depot and the arc exists in the pruned x variables
                    if i != 0 and (l, i, j, t_bar) in loader_flow.x
                ]

                # Sum over those existing arcs.
                cumulative_flow = quicksum(loader_flow.x[arc] for arc in valid_arcs)

                # Add indicator constraints - now referencing an existing started[l, j, t]
                m.addGenConstrIndicator(
                    loader_flow.started[l, j, t], True,
                    cumulative_flow, GRB.GREATER_EQUAL, 1.0 - epsilon,  # Use tolerance
                    name=f"ind_start_1_{l}_{j}_{t}"
                )

                m.addGenConstrIndicator(
                    loader_flow.started[l, j, t], False,
                    cumulative_flow, GRB.LESS_EQUAL, 0.0 + epsilon,  # Use tolerance
                    name=f"ind_start_0_{l}_{j}_{t}"
                )
        else:
            print("Skipping 'Started' indicator constraints as no 'started' variables exist.")

        # Define Finished: Link finished variable to remaining beet volume at the location
        # Iterate only over the keys for which the 'finished' variable exists

        print("Adding 'Finished' definition constraints...")
        if loader_flow.finished:  # Check if the dictionary is not empty
            for (l, i, t) in loader_flow.finished.keys():
                # The key (l, i, t) guarantees i != 0 here

                # Ensure the corresponding beet flow variable exists (should generally exist)
                # Note: BeetFlow 'b' indices are (field, source, sink, time)
                beet_var_key = (i, 0, 0, t)  # Volume remaining at field i (source=0) at heap (sink=0) at time t
                if beet_var_key not in beet_flow.b:
                    print(
                        f"Warning: Beet variable {beet_var_key} not found for Finished constraint ({l},{i},{t}). "
                        f"Skipping.")
                    continue

                # Add constraints - now referencing an existing finished[l, i, t]
                # If finished=1, beet_volume must be near 0.
                # If finished=0, beet_volume can be > 0.
                m.addConstr(
                    beet_flow.b[i, 0, 0, t] <= (beet_volume[i] + epsilon) * (
                            1 - loader_flow.finished[l, i, t]),
                    name=f"finish_upper_{l}_{i}_{t}"
                )
                # This lower bound might be too strict if epsilon is small and flow is continuous?
                # Maybe remove if causing issues, or adjust epsilon.
                # It tries to force finished=0 if *any* volume > epsilon remains.
                m.addConstr(
                    beet_flow.b[i, 0, 0, t] >= epsilon * (1 - loader_flow.finished[l, i, t]),
                    name=f"finish_lower_{l}_{i}_{t}"
                )
        else:
            print("Skipping 'Finished' definition constraints as no 'finished' variables exist.")

    #
    # IDLE Constraints (split x into idle and working)
    #

    loader_flow.add_idle(v_type=v_type)

    #
    # PARTIAL WORK (Y)
    #

    loader_flow.add_simple_travel_time()

    #
    # CAPACITY CONSTRAINTS
    #

    # BEET FLOW dependent on Y
    for l in L:
        for i in Il[l]:
            if i != 0:
                for t in T:
                    # Sum over valid outgoing arcs from i:
                    out_sum = quicksum(
                        L_bar[l] * loader_flow.y_out[l, i, j, t]
                        for j in Il[l] if (l, i, j, t) in loader_flow.x
                    )
                    # Sum over valid incoming arcs to i:
                    in_sum = quicksum(
                        L_bar[l] * loader_flow.y_in[l, j, i, t]
                        for j in Il[l] if (l, j, i, t) in loader_flow.x
                    )
                    m.addConstr(
                        (beet_flow.b[i, 0, 1, t] + beet_flow.b[i, 0, 2, t]) <= (out_sum + in_sum),
                        f"loader_fractional_cap_simple_i{i}_t{t}"
                    )

    # Inventory Capacity
    print("Inventory Capacity: ", IC)
    for t in T:
        m.addConstr(
            quicksum(beet_flow.b[i, 1, 1, t] for i in I) <= IC,
            f"inventory_capacity")

    # Production Capacity

    # Maximum production demand constraint with unmet demand
    for t in T:
        #if t > 0:
        m.addConstr(
            quicksum(beet_flow.b[i, k, 2, t] for i in I for k in range(0, 2)) <=
            P_bar_max[t],
            f"max_production{t}"
        )

    # Min production demand constraint with unmet demand
    for t in T:
        #if t > 0:
        m.addConstr(
            quicksum(beet_flow.b[i, k, 2, t] for i in I for k in range(0, 2)) + unmet_demand[t] >=
            P_bar_min[t],
            f"min_production{t}"
        )

    # CONDITIONAL

    #
    # Coarse or normal Hotstart or Enforce
    #

    # --- Apply Warm - start or Enforce Solution - --
    if coarse_hotstart_schedule and coarse_hotstart_config:
        hs_type = coarse_hotstart_config.get('type')
        # ... (your existing 'initial_fine_movements' logic if kept) ...
        if hs_type == 'initial_fine_movements':
            num_steps = coarse_hotstart_config.get('num_steps', 0)
            if num_steps > 0:
                print(f"Applying 'initial_fine_movements' warm-start for {num_steps} fine steps.")
                # Call your previous function:
                from_coarse_schedule_to_initial_fine_hot_start(m, loader_flow, coarse_hotstart_schedule, Il, T,
                                                               num_steps)
                pass
        elif hs_type == 'selective_depot_related':  # New type
            print(  # Using name if combined_key not available here
                f"Applying 'selective_depot_related' warm-start for {name}.")
            ratio_for_hotstart = coarse_hotstart_config.get('ratio')
            if ratio_for_hotstart and isinstance(ratio_for_hotstart, int) and ratio_for_hotstart > 0:
                from_coarse_schedule_to_selective_fine_hot_start(
                    m, loader_flow, coarse_hotstart_schedule, Il,
                    T, ratio_for_hotstart, coarse_hotstart_config  # Pass the whole config dict for flags
                )
            else:
                print(
                    f"Warning: 'ratio' missing or invalid in coarse_hotstart_config for selective_depot_related type. "
                    f"Ratio found: {ratio_for_hotstart}")
        else:
            print(f"Warning: Unknown coarse_hotstart_config type: {hs_type}")

    elif hotstart_solution:  # Existing full fine-grained hotstart from heuristic/other
        # Check if scenario key is needed
        hs_schedule_data = hotstart_solution
        if combined_key and combined_key in hotstart_solution:  # If hotstart_solution is keyed by combined_key
            hs_schedule_data = hotstart_solution[combined_key]

        if "loader_schedule" in hs_schedule_data:
            loader_schedule = hs_schedule_data["loader_schedule"]
            print("Applying standard hotstart from provided fine-grained schedule.")
            print(f"Hotstart Schedule: {loader_schedule}")
            # Assuming from_schedule_to_hot_start_pruned is the correct one to use here
            # You might need to adapt this if model_version['idle_time'] is relevant for choosing _idle vs _pruned
            from_schedule_to_hot_start_pruned(m, loader_flow, loader_schedule, Il, T)
        else:
            print("Warning: hotstart_solution provided but 'loader_schedule' key is missing.")

    if enforce_solution:
        # ... (existing enforce_solution logic) ...
        # check if scenario key is needed
        enf_schedule_data = enforce_solution
        if combined_key and combined_key in enforce_solution:
            enf_schedule_data = enforce_solution[combined_key]

        if "loader_schedule" in enf_schedule_data:
            loader_schedule = enf_schedule_data["loader_schedule"]
            print("Enforcing schedule from provided fine-grained schedule.")
            print(f"Enforce Schedule: {loader_schedule}")
            # Assuming enforce_schedule_with_constraints_pruned is appropriate
            enforce_schedule_with_constraints_pruned(m, loader_flow, loader_schedule, Il, T)
        else:
            print("Warning: enforce_solution provided but 'loader_schedule' key is missing.")

    #
    # Extra constraints
    #

    # If loader at depot, working = 1 (to avoid penalty)
    for l in L:
        for t in T:
            m.addConstr(loader_flow.working[l, 0, t] == loader_flow.x[l, 0, 0, t],
                        f"working_in_depot{l}_{t}"
                        )

    # Optional: vehicle capacity
    if vehicle_capacity_flag:

        print("Add Vehicle Capacity Constraint")
        # Vehicle Capacity
        for l in L:
            m.addConstr(
                quicksum(beet_flow.b[i, 0, k, t] for i in Il[l] if i != 0 for k in range(1, 3) for t in T) <=
                vehicle_cap[l],
                f"vehicle_capacity{l}"
            )

            # m.addConstr(
            #    quicksum(beet_flow.b[i, 0, k, t] for i in Il[l] if i != 0 for k in range(1, 3) for t in T) >=
            #    0.95 * vehicle_cap[l],
            #    f"vehicle_capacity{l}"
            # )

        # Inventory goals and vehicle capacity are currently cross dependent, change from bool to float if needed
        if inventory_flag:
            if sum(beet_volume[i] for l in L for i in Il[l] if i != 0) >= production_volume_per_day * 5:
                if inventory_goals < 0:
                    raise ValueError("inventory_goals below 0 in solve_loader_flow!")

                print("Add Inventory Goals in Solver: ", inventory_goals)
                m.addConstr(quicksum(beet_flow.b[i, 1, 1, max(T)] for i in I) >= inventory_goals, "inventory_min")
                if inventory_cap_flag:
                    print(f"Inventory Capacity Limit Activated: {inventory_goals + 500}")
                    m.addConstr(quicksum(beet_flow.b[i, 1, 1, max(T)] for i in I) <= inventory_goals + 500,
                                "inventory_cap")
                else:
                    print("Inventory No Capacity Limits")

    else:  # if we have vehicle capacity we dont need to finish all fields and vice versa (may need to ind. flag)
        print("Add Finish all Fields Constraint (vehicle cap = None)")
        m.addConstr(quicksum(beet_flow.b[i, 0, 0, max(T)] for i in I) <= 0, "load_all_fields")

    if holidays_flag:
        print("Add Holiday Constraint")
        loader_flow.add_holiday_constraint()

    #
    # ******* Additional Pruning and Variables Setting (non-standard) *******
    #

    # In solve_loader_flow, after beet_flow and loader_flow are initialized
    # and loader_flow.create_decision_variables() has run (which calculates ESTs)

    # Assuming loader_flow stores the ESTs in a way accessible like:
    # loader_flow.loader_earliest_start_time_at_node[loader_id][field_id]

    # In your solve_loader_flow function, after beet_flow and loader_flow are initialized
    # and loader_flow.create_decision_variables() has successfully populated loader_flow.x

    print("Applying direct loader activity-based pruning to BeetFlow movement variables...")

    # First, determine which (field, time) pairs have any potential loader activity
    # based on the existing loader_flow.x variables.
    # Fields are unique to loaders.
    active_field_times = defaultdict(set)  # Stores field_i -> {t1, t2, ...} where activity is possible

    if loader_flow.x:  # Check if loader_flow.x variables were created
        for (l_id, from_loc, to_loc, t_period) in loader_flow.x.keys():
            # If the loader is departing from a field (that isn't depot), that field is active at t_period
            if from_loc != 0 and from_loc in loader_flow.Il.get(l_id, []):  # Ensure from_loc is a field for this loader
                active_field_times[from_loc].add(t_period)
            # If the loader is arriving at a field (that isn't depot), that field is active at t_period
            if to_loc != 0 and to_loc in loader_flow.Il.get(l_id, []):  # Ensure to_loc is a field for this loader
                active_field_times[to_loc].add(t_period)
    else:
        print("Warning: No loader_flow.x variables exist. BeetFlow pruning based on loader activity cannot be applied.")

    # Now, iterate through beet flow variables and prune if no loader activity is possible

    if beet_flow.b:  # Check if beet_flow.b variables exist

        changed_beet_flow_vars_count = 0

        for field_i in beet_flow.I:  # beet_flow.I should be your set of fields (non-depot)
            if field_i == 0:
                continue  # Should not happen if beet_flow.I is just fields

            # Get the set of active times for this specific field_i
            possible_activity_times_for_field = active_field_times.get(field_i, set())

            for t_period in T:  # Iterate through all time periods in the model
                if t_period not in possible_activity_times_for_field:
                    # If t_period is NOT a time where any loader x-variable touches field_i,
                    # then no y_in/y_out can be generated for this field at this time.
                    # So, prune beet movements from the field heap.

                    # Prune b[field_i, 0, 1, t_period] (field heap to harvested/storage)
                    key_heap_to_harvested = (field_i, 0, 1, t_period)
                    if key_heap_to_harvested in beet_flow.b:
                        if beet_flow.b[key_heap_to_harvested].ub > 1e-7:  # Check if UB is effectively > 0
                            beet_flow.b[key_heap_to_harvested].ub = 0.0
                            changed_beet_flow_vars_count += 1
                            # print(f"Pruning: Setting UB of b[{field_i},0,1,{t_period}] to 0 (no loader activity)")

                    # Prune b[field_i, 0, 2, t_period] (field heap to production - for load_only_flow)
                    key_heap_to_production = (field_i, 0, 2, t_period)
                    if key_heap_to_production in beet_flow.b:  # This check handles if (0,2) is an allowed transition
                        if beet_flow.b[key_heap_to_production].ub > 1e-7:  # Check if UB is effectively > 0
                            beet_flow.b[key_heap_to_production].ub = 0.0
                            changed_beet_flow_vars_count += 1
                            # print(f"Pruning: Setting UB of b[{field_i},0,2,{t_period}] to 0 (no loader activity)")

        # Verbose
        if changed_beet_flow_vars_count > 0:
            print(
                f"Pruning: Set Upper Bound to 0 for {changed_beet_flow_vars_count} beet flow movement variables due to "
                f"lack of corresponding loader activity.")
        else:
            print(
                "No beet flow movement variables were further pruned by setting UB to 0 based on loader activity "
                "(they might have been already 0 or no such cases found).")

        m.update()  # Apply bound changes to the model
    else:
        print("Warning: No beet_flow.b variables exist. Pruning cannot be applied.")

    #
    # OBJECTIVE FUNCTION
    #

    # Objective: Maximize Profit
    m.setObjective(

        # Revenue component: Beet produced from fields (both production & stored) at final time period.
        quicksum((beet_flow.b[i, 2, 2, max(T)] + beet_flow.b[i, 1, 1, max(T)]) for i in I) * BEET_PRICE

        - quicksum(  # Subtract Travel Cost:
            loader_flow.c_l[l].at[i, j] * loader_flow.x[l, i, j, t]
            for (l, i, j, t) in loader_flow.x.keys() if i != j
        )
        - quicksum(  # Subtract Operating Cost:
            loader_flow.c_l[l].at[i, i] *
            (loader_flow.working[l, i, t])  # we need to adapt fot idle keys
            # Unpack key directly into (l, i, t)
            for (l, i, t) in loader_flow.working.keys() if i != 0 and t > 0
        )

        - quicksum(  # Subtract idle Cost:
            loader_flow.c_l[l].at[i, i] *
            (LAMBDA * loader_flow.idle[l, i, t])  # we need to adapt fot idle keys
            # Unpack key directly into (l, i, t)
            for (l, i, t) in loader_flow.working.keys() if i != 0 and t > 0 and t not in holidays
        )

        - quicksum(  # Subtract Inventory Costs:
            c_s * beet_flow.b[i, 1, 1, t]
            for i in I for t in T
        )
        - quicksum(  # Subtract Unmet Demand Penalty:
            penalty_rate * unmet_demand[t]
            for t in T
        ),
        GRB.MAXIMIZE
    )

    # --- Set solver parameters & optimize -------------------------------------------------
    for p, v in FIXED_SOLVER_PARAMS.items():
        m.setParam(p, v)

    print("\nStart Optimisation Model Version: ", combined_key, "\n")

    m.optimize()

    # Initialize the results dictionary
    scenario_results = {}

    # Handle different optimization statuses
    if m.status == GRB.INFEASIBLE:
        print("m is infeasible. Computing IIS to find the issue...")
        m.computeIIS()
        m.write("m_iis.ilp")
        print(f"IIS written to file 'm_iis.ilp'")
        print("Review the IIS file to identify conflicting constraints.")
        scenario_results['status'] = 'infeasible'
    elif m.status == GRB.OPTIMAL:
        scenario_results['status'] = 'best_solution'
    elif m.status == GRB.SUBOPTIMAL:
        scenario_results['status'] = 'suboptimal_solution'
    else:
        print(f"m status: {m.status}")
        scenario_results['status'] = 'no_solution'

    # Save params
    parameters = {
        "c_l": c_l,
        "L": L,
        "T": T,
        "I": I,
        "Il": Il,
        "L_bar": L_bar,
        "holidays": holidays,
        "operations_cost_t": c_l[next(iter(L))].iloc[1, 1],
        # Get any cost of a loader (they are currently all the same)
        "sugar_price": BEET_PRICE,
        "sugar_concentration": 0.15,
        "tau": tau,
        "t_p": t_p,
        "MIP_gap": m.MIPGap,
        "runtime": m.Runtime,
        "c_s_t": c_s,
        "BEET_PRICE": BEET_PRICE,
        "UD_PENALTY": penalty_rate,
        "LAMBDA": LAMBDA
    }
    scenario_results["parameters"] = parameters

    # Save variables
    decision_variables = {
        # Standard Variables
        'loader_flow_x': {k: v.X for k, v in loader_flow.x.items() if abs(v.X) > 1e-6},  # Save memory storing only 1s
        'beet_flow_b': {k: v.X for k, v in beet_flow.b.items()},  # Interested in 0 values
        'unmet_demand': {k: v.X for k, v in unmet_demand.items()},
        'working': {k: v.X for k, v in loader_flow.working.items() if abs(v.X) > 1e-6},
        'idle': {k: v.X for k, v in loader_flow.idle.items() if abs(v.X) > 1e-6},
        'y_in': {k: v.X for k, v in loader_flow.y_in.items() if abs(v.X) > 1e-6},
        'y_out': {k: v.X for k, v in loader_flow.y_out.items() if abs(v.X) > 1e-6},
        # Restricted Variables are Conditional
        'started': {k: v.X for k, v in loader_flow.started.items()} if loader_flow.started else None,
        'finished': {k: v.X for k, v in loader_flow.finished.items()} if loader_flow.finished else None,
        'restricted': {k: v.X for k, v in loader_flow.restricted.items()} if loader_flow.restricted else None,
    }
    scenario_results["decision_variables"] = decision_variables

    # Run Post Solution Check
    if last_period_restricted_flag:
        loading_flow_solution_check(params=parameters, decision_variables=decision_variables)

    else:
        print("WARNING: No Post-Solution Check because last_period_restricted_flag is deactivated")

    # Save Results
    results_file_path = os.path.join(base_file_path, f"results/reporting/results_{name}.pkl")

    with open(results_file_path, 'wb') as results_file:
        pickle.dump(scenario_results, results_file)

    print(f"Sav file to: {results_file_path}")
    print(f"All results of {name} saved as pickle")

    return scenario_results


def run_loader_flow_experiments(
        scenarios,
        instance_data,
        sensitivity_scenarios,
        gurobi_model_versions,
        solver_params,
        base_file_path,
        experiment_name,
        hotstart_solution=None,
        enforce_solution=None,
        vehicle_capacity_flag=False,
        restricted_flow_flag=True,
        last_period_restricted_flag=False,
        time_restriction=False,
        custom_T=None,
        verbose=False,
        v_type="binary"):
    # Create empty dic to store results
    all_results = {}

    print("\nStart Inside Gurobi Function Heuristic:\n")
    # Create scenario results to get a T and delay estimation
    heuristic_results = run_pl_heuristic_experiments(scenarios, instance_data, sensitivity_scenarios, base_file_path,
                                                     model_versions=gurobi_model_versions,
                                                     vehicle_capacity_flag=vehicle_capacity_flag,
                                                     time_restriction=time_restriction,
                                                     verbose=True,
                                                     usage=True)

    if verbose:
        print("Finished heuristic")

    for size_scn in scenarios:

        # Extract a unique scenario identifier (e.g., using fields, harvesters, and loaders)
        scenario_id = f"{size_scn['nr_fields']}_{size_scn['nr_h']}_{size_scn['nr_l']}"

        scenario_instance_data = instance_data[scenario_id]

        print("START SIZE SCENARIO: ", scenario_id)

        for model_ver in gurobi_model_versions:

            for sens_scn in sensitivity_scenarios:
                # Unique identifier for storing results
                combined_key = make_key(size_scn, model_ver, sens_scn)

                # Prepare Data:
                print(heuristic_results.keys())
                if custom_T:
                    print("Warning: Use Custom T")
                    T_heuristic = custom_T
                else:
                    T_heuristic = heuristic_results[combined_key]["T"]
                loader_routes_heuristic = heuristic_results[combined_key]["loader_routes"]

                instance_raw, derived = prepare_loader_flow_data(scenario_instance_data,
                                                                 sens=sens_scn,
                                                                 model_ver=model_ver,
                                                                 T_heuristic=T_heuristic,
                                                                 loader_routes_heuristic=loader_routes_heuristic)

                # Solve Model
                scenario_results = solve_loader_flow(
                    instance=instance_raw,
                    derived=derived,
                    name="test",
                    hotstart_solution=hotstart_solution,  # given schedule a nested list
                    enforce_solution=enforce_solution,  # given schedule a nested list
                    vehicle_capacity_flag=vehicle_capacity_flag,
                    restricted_flow_flag=restricted_flow_flag,
                    last_period_restricted_flag=last_period_restricted_flag,
                    FIXED_SOLVER_PARAMS=solver_params,
                    verbose=False,
                    combined_key=combined_key,
                    base_file_path=base_file_path,
                    v_type=v_type)

                # Save the scenario results in all_results with the combined unique key
                all_results[combined_key] = scenario_results

    # Save with date id
    today_date = datetime.now().strftime('%Y%m%d')
    results_file_path = os.path.join(base_file_path,
                                     f"results/reporting/results_{experiment_name}.pkl")
    with open(results_file_path, 'wb') as results_file:
        pickle.dump(all_results, results_file)
    print(f"Sav file to: {results_file_path}")
    print(f"All results of {experiment_name} saved as pickle")

    return all_results


def run_loader_flow_TimeAgg(
        scenarios: List[Dict[str, Any]],
        instance_data2h: Dict[str, Any],
        instance_data1h: Dict[str, Any],
        sensitivity_scenarios: List[Dict[str, Any]],
        gurobi_model_versions_2h: List[Dict[str, Any]],
        gurobi_model_versions_1h: List[Dict[str, Any]],
        solver_params: Dict[str, Any],
        base_file_path: str,
        experiment_name: str,
        time_agg_warmstart_params: Optional[Dict[str, Any]] = None,
        hotstart_solution: Optional[Dict[str, Any]] = None,
        enforce_solution: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Orchestrates the solution of the Multi-Commodity Flow Harvest Planning Model
    across different time aggregations (e.g., 2-hour and 1-hour periods),
    leveraging a coarse-to-fine warm-start approach.

    This function first runs a heuristic for both 2-hour and 1-hour models
    to get initial time horizon estimates and routes. Then, it solves the 2-hour
    model, extracts its solution (including the coarse schedule and remaining
    beet volumes), and uses this information to warm-start and prune the 1-hour model.

    Args:
        scenarios (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
            defines a scenario (e.g., number of fields, harvesters, loaders).
        instance_data2h (Dict[str, Any]): A dictionary containing instance data
            pre-processed for the 2-hour time aggregation. Keys typically correspond
            to scenario IDs.
        instance_data1h (Dict[str, Any]): A dictionary containing instance data
            pre-processed for the 1-hour time aggregation. Keys typically correspond
            to scenario IDs.
        sensitivity_scenarios (List[Dict[str, Any]]): A list of dictionaries, each
            defining a sensitivity scenario (e.g., cost multipliers, productivity multipliers).
        gurobi_model_versions_2h (List[Dict[str, Any]]): A list of dictionaries, each
            specifying model version parameters for the 2-hour models (e.g., idle time,
            restricted access, working hours, travel time, time period length 't_p').
        gurobi_model_versions_1h (List[Dict[str, Any]]): A list of dictionaries, each
            specifying model version parameters for the 1-hour models.
        solver_params (Dict[str, Any]): A dictionary of Gurobi solver parameters
            to be applied (e.g., MIPGap, TimeLimit).
        base_file_path (str): The base directory path for loading and saving data.
        experiment_name (str): A unique name for the current experiment, used for
            naming the results file.
        time_agg_warmstart_params (Optional[Dict[str, Any]]): Configuration for
            the time aggregation warm-start for the 1-hour model, detailing the type
            of warm-start (e.g., 'selective_depot_related') and specific flags
            (e.g., 'set_initial_depot_stays', 'ratio'). Defaults to None.
        hotstart_solution (Optional[Dict[str, Any]]): An optional pre-computed
            fine-grained schedule (e.g., from a previous run or a heuristic) to be
            used as a direct warm-start for any of the models. Defaults to None.
        enforce_solution (Optional[Dict[str, Any]]): An optional pre-computed
            fine-grained schedule to be enforced as hard constraints in any of the
            models. Defaults to None.
        params (Optional[Dict[str, Any]]): An optional dictionary to override or
            extend the default parameters for model flags like vehicle capacity,
            restricted flow, etc. Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary containing all scenario results, where keys are
            combined scenario identifiers (e.g., "nr_fields_nr_h_nr_l_MV_..._S_...")
            and values are the detailed results dictionaries from `solve_loader_flow`.
    """

    # Set default parameters if not provided or merge with provided ones
    _default_params = {
        'vehicle_capacity_flag': False,
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


    # Create empty dic to store results
    all_results = {}

    print("\nStart Inside Gurobi Function Heuristic:\n")
    # Create scenario results to get a T and delay estimation

    print("\n2h Heuristic:\n")
    heuristic_results2h = run_pl_heuristic_experiments(scenarios, instance_data2h, sensitivity_scenarios,
                                                       base_file_path,
                                                       model_versions=gurobi_model_versions_2h,
                                                       vehicle_capacity_flag=vehicle_capacity_flag,
                                                       time_restriction=time_restriction,
                                                       verbose=True,
                                                       usage=True)

    print("\n1h Heuristic:\n")
    heuristic_results1h = run_pl_heuristic_experiments(scenarios, instance_data1h, sensitivity_scenarios,
                                                       base_file_path,
                                                       model_versions=gurobi_model_versions_1h,
                                                       vehicle_capacity_flag=vehicle_capacity_flag,
                                                       time_restriction=time_restriction,
                                                       verbose=True,
                                                       usage=True)

    if verbose:
        print("Finished heuristic")

    for size_scn in scenarios:

        # Extract a unique scenario identifier (e.g., using fields, harvesters, and loaders)
        scenario_id = f"{size_scn['nr_fields']}_{size_scn['nr_h']}_{size_scn['nr_l']}"

        scenario_instance_data_2h = instance_data2h[scenario_id]
        scenario_instance_data_1h = instance_data1h[scenario_id]

        print("\nSTART SIZE SCENARIO: ", scenario_id, "\n")

        for model_ver, model_ver_2h in enumerate(gurobi_model_versions_2h):

            for sens_scn in sensitivity_scenarios:

                # Unique identifier for storing results
                combined_key_1h = make_key(size_scn, gurobi_model_versions_1h[model_ver], sens_scn)
                combined_key_2h = make_key(size_scn, gurobi_model_versions_2h[model_ver], sens_scn)

                # model_ver_1h = gurobi_model_versions_1h[model_ver]
                model_ver_2h = gurobi_model_versions_2h[model_ver]

                # Prepare Data:

                T_heuristic_2h = heuristic_results2h[combined_key_2h]["T"]
                loader_routes_heuristic = heuristic_results2h[combined_key_2h]["loader_routes"]

                instance_raw_2h, derived_2h = prepare_loader_flow_data(scenario_instance_data_2h,
                                                                       sens=sens_scn,
                                                                       model_ver=model_ver_2h,
                                                                       T_heuristic=T_heuristic_2h,
                                                                       loader_routes_heuristic=loader_routes_heuristic)

                # Solve Model 2h
                scenario_results_2h = solve_loader_flow(
                    instance=instance_raw_2h,
                    derived=derived_2h,
                    name="2h_Single_Model",
                    hotstart_solution=hotstart_solution,  # given schedule a nested list
                    enforce_solution=enforce_solution,  # given schedule a nested list
                    vehicle_capacity_flag=vehicle_capacity_flag,
                    restricted_flow_flag=restricted_flow_flag2h,
                    last_period_restricted_flag=last_period_restricted_flag2h,
                    # add_min_beet_restriction_flag=add_min_beet_restriction_flag2h, # Usually not needed
                    FIXED_SOLVER_PARAMS=solver_params,
                    verbose=False,
                    combined_key=combined_key_2h,
                    base_file_path=base_file_path,
                    v_type=v_type)

                # Extract Gurobi's decision variables from the 2h model solution
                decision_variables_2h = scenario_results_2h["decision_variables"]
                parameters_2h = scenario_results_2h["parameters"]

                # Extract last period beets
                last_period_beet_volume = extract_last_period_beet_volume(
                    params=parameters_2h,
                    decision_variables=decision_variables_2h)

                # Extract Coarse Schedule from 2h model for potential warmstart of 1h model
                coarse_schedule_for_1h_hotstart = None

                if decision_variables_2h and 'loader_flow_x' in decision_variables_2h and decision_variables_2h[
                    'loader_flow_x']:
                    coarse_schedule_for_1h_hotstart = extract_loader_schedule_list(decision_variables_2h)
                    # print(f"  Coarse schedule from 2h model: {coarse_schedule_for_1h_hotstart}") # For debugging
                else:
                    print(
                        f"  Warning: No valid decision variables from 2h model to extract coarse schedule for "
                        f"{combined_key_2h}.")

                # Generate valid_combinations for pruning the 1h model (existing logic)
                valid_combinations = generate_valid_arcs_for_all_loaders(
                    decision_variables=decision_variables_2h,  # Use 2h solution
                    ratio=2,  # Assuming 2h to 1h ratio for now
                    buffer_pct=derived_2h["buffer_pct"]  # Example buffer
                )

                # --- CHANGE: Determine required T for 1h model ---
                max_tf_from_combinations = -1  # Initialize to -1 (relevant if valid_combinations is empty)
                if valid_combinations:
                    # Iterate through each loader's time->arcs dictionary
                    for loader_id in valid_combinations:
                        if valid_combinations[loader_id]:  # Check if inner dict is not empty
                            # Find the maximum time key (t_f) present for this loader
                            max_tf_loader = max(valid_combinations[loader_id].keys())
                            # Keep track of the overall maximum time step found
                            max_tf_from_combinations = max(max_tf_from_combinations, max_tf_loader)

                # The required number of time steps is max_tf + 1
                # If no combinations were found (max_tf_from_combinations is still -1), use 0.
                T_length_from_combinations = max_tf_from_combinations + 1 if max_tf_from_combinations >= 0 else 0

                # Get the heuristic T length for the 1h scenario
                T_length_heuristic_1h = heuristic_results1h[combined_key_1h]["T"]

                # Use the maximum of the heuristic length and the length required by the valid combinations
                T_final_1h = max(T_length_heuristic_1h, T_length_from_combinations)

                print(f"  Determined T Horizon for 1h Model:")
                print(f"    Heuristic Suggestion (T_heuristic_1h): {T_length_heuristic_1h}")
                print(
                    f"    Max Time Index from Valid Combinations: {max_tf_from_combinations} "
                    f"(Required Length: {T_length_from_combinations})")
                print(f"    ==> Using Final T Horizon: {T_final_1h}")

                model_ver_1h_actual = gurobi_model_versions_1h[model_ver]  # Get the actual 1h model_ver

                calculated_ratio = int(model_ver_2h.get("t_p", 2) / model_ver_1h_actual.get("t_p", 1))

                instance_raw_1h, derived_1h = prepare_loader_flow_data(scenario_instance_data_1h,
                                                                       sens=sens_scn,
                                                                       model_ver=model_ver_1h_actual,
                                                                       T_heuristic=T_final_1h,
                                                                       loader_routes_heuristic=None,
                                                                       valid_arcs_at_time=valid_combinations,
                                                                       last_period_beet_volume=last_period_beet_volume)

                # --- Configure Time Aggregation Warmstart for 1h model ---
                current_coarse_hotstart_config_arg = None
                if time_agg_warmstart_params and coarse_schedule_for_1h_hotstart:
                    ws_type = time_agg_warmstart_params.get('type')
                    if ws_type == 'selective_depot_related':
                        current_coarse_hotstart_config_arg = {
                            'type': 'selective_depot_related',
                            'ratio': calculated_ratio,  # Crucial: t_p_coarse / t_p_fine
                            'set_initial_depot_stays': time_agg_warmstart_params.get('set_initial_depot_stays', True),
                            'set_initial_depot_egress': time_agg_warmstart_params.get('set_initial_depot_egress', True),
                            'set_terminal_depot_stays': time_agg_warmstart_params.get('set_terminal_depot_stays', True),
                        }
                        print(
                            f"  Configured selective depot-related warmstart for 1h model with "
                            f"ratio {calculated_ratio}.")
                    # else:
                    # print(f"  Time_agg warmstart not applied: num_steps is 0 or not specified for {combined_key_1h}.")
                # elif not coarse_schedule_for_1h_hotstart:
                # print(f"  Time_agg warmstart not applied: Coarse schedule from 2h model is missing for {combined_key_1h}.")

                # Solve Model 1h (fine model)
                scenario_results_1h = solve_loader_flow(
                    instance=instance_raw_1h,
                    derived=derived_1h,
                    name="1h_Single_TimeAgg",  # Changed name
                    hotstart_solution=None,  # Standard hotstart for 1h is None if using the new mechanism
                    enforce_solution=None,  # Standard enforce for 1h is None
                    vehicle_capacity_flag=vehicle_capacity_flag,
                    restricted_flow_flag=restricted_flow_flag1h,
                    last_period_restricted_flag=last_period_restricted_flag1h,
                    add_min_beet_restriction_flag=add_min_beet_restriction_flag1h,
                    coarse_hotstart_schedule=coarse_schedule_for_1h_hotstart,
                    coarse_hotstart_config=current_coarse_hotstart_config_arg,
                    FIXED_SOLVER_PARAMS=solver_params,
                    verbose=False,  # Or pass outer verbose
                    combined_key=combined_key_1h,
                    base_file_path=base_file_path,
                    v_type=v_type)

                # Save the 2h and 1h with identifier
                all_results[combined_key_1h] = {
                    "opt_type": "time_agg",
                    "scenario_results_coarse": scenario_results_2h,
                    "scenario_results_fine": scenario_results_1h
                }

    # Save with date id
    results_file_path = os.path.join(base_file_path,
                                     f"results/reporting/results_{experiment_name}.pkl")
    with open(results_file_path, 'wb') as results_file:
        pickle.dump(all_results, results_file)

    print(f"Save file to: {results_file_path}")
    print(f"All results of {experiment_name} saved as pickle")

    return all_results

def solve_time_agg_loader_flow(
    # --- Data & Config Inputs ---
    instance_raw_coarse: Dict[str, Any],
    instance_raw_fine: Dict[str, Any],
    model_ver_coarse: Dict[str, Any],
    model_ver_fine: Dict[str, Any],
    sens_scn: Dict[str, Any],
    solver_params: Dict[str, Any],
    # --- Heuristic Inputs (pre-calculated) ---
    T_heuristic_coarse: int,
    T_heuristic_fine: int,
    loader_routes_heuristic: Dict[int, List[int]],
    # --- Hotstart 2h version
    hotstart_solution2h= None,
    enforce_solution2h= None,
    # --- Time Aggregation Specific Params ---
    time_agg_warmstart_params: Optional[Dict[str, Any]] = None,
    # --- General Model & Output Params ---
    base_file_path: str = "../../data/",
    combined_key: str = "time_agg_run",
    name: str = "test",
    # --- Coarse Model Flags ---
    vehicle_capacity_flag_coarse: bool = False,
    restricted_flow_flag_coarse: bool = True,
    last_period_restricted_flag_coarse: bool = False,
    add_min_beet_restriction_flag_coarse: bool = False,
    # --- Fine Model Flags ---
    vehicle_capacity_flag_fine: bool = False,
    restricted_flow_flag_fine: bool = True,
    last_period_restricted_flag_fine: bool = False,
    add_min_beet_restriction_flag_fine: bool = False,
    # --- Common Flags ---
    inventory_flag=True,
    inventory_cap_flag=False,
    v_type: str = "binary",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Solves a single instance of the loader flow problem using a coarse-to-fine
    time aggregation strategy.

    This function first solves a coarse model, then uses its results (schedule,
    final beet volumes) to prune and warm-start a fine model. It is designed to
    be called for a single scenario, separate from experimental loops.

    Args:
        instance_raw_coarse: Raw instance data for the coarse model.
        instance_raw_fine: Raw instance data for the fine model.
        model_ver_coarse: Model version parameters for the coarse model.
        model_ver_fine: Model version parameters for the fine model.
        sens_scn: Sensitivity scenario parameters.
        solver_params: Gurobi solver parameters.
        T_heuristic_coarse: Time horizon from heuristic for the coarse model.
        T_heuristic_fine: Time horizon from heuristic for the fine model.
        loader_routes_heuristic: Heuristically determined routes for loaders.
        time_agg_warmstart_params: Configuration for the warm-start process.
        base_file_path: Base directory for saving files.
        combined_key: A unique key for this specific run.
        *_flag_coarse: Constraint flags for the coarse model.
        *_flag_fine: Constraint flags for the fine model.
        v_type: Gurobi variable type (e.g., "binary").
        verbose: Enables verbose logging.

    Returns:
        A dictionary containing the results of the final (fine) optimization run,
        in the same format as solve_loader_flow.
    """
    print(f"\n--- Starting Time Aggregation for: {combined_key} ---\n")
    scenario_results = {}
    #
    # === STEP 1: SOLVE COARSE (2h) MODEL ===
    #
    print("1. Preparing Coarse Model Data...")
    instance_raw_2h, derived_2h = prepare_loader_flow_data(
        instance_raw_coarse,
        sens=sens_scn,
        model_ver=model_ver_coarse,
        T_heuristic=T_heuristic_coarse,
        loader_routes_heuristic=loader_routes_heuristic
    )

    print("2. Solving Coarse Model...")
    scenario_results_2h = solve_loader_flow(
        instance=instance_raw_2h,
        derived=derived_2h,
        name=f"{combined_key}_coarse",
        hotstart_solution=hotstart_solution2h,
        enforce_solution=enforce_solution2h,
        vehicle_capacity_flag=vehicle_capacity_flag_coarse,
        restricted_flow_flag=restricted_flow_flag_coarse,
        last_period_restricted_flag=last_period_restricted_flag_coarse,
        add_min_beet_restriction_flag=add_min_beet_restriction_flag_coarse,
        FIXED_SOLVER_PARAMS=solver_params,
        verbose=verbose,
        combined_key=combined_key,
        base_file_path=base_file_path,
        inventory_flag=inventory_flag,
        inventory_cap_flag=inventory_cap_flag,
        v_type=v_type
    )

    #
    # === STEP 2: PROCESS COARSE RESULTS TO PREPARE FOR FINE MODEL ===
    #
    print("\n3. Processing Coarse Model Results...")
    if not scenario_results_2h["decision_variables"]:
        print("  ERROR: Coarse model was infeasible or failed to solve. Halting.")
        return scenario_results_2h  # Return the failure result

    decision_variables_2h = scenario_results_2h["decision_variables"]
    parameters_2h = scenario_results_2h["parameters"]

    # Extract final beet levels to set as a goal for the fine model
    last_period_beet_volume = extract_last_period_beet_volume(
        params=parameters_2h,
        decision_variables=decision_variables_2h
    )
    print(f"  Extracted final beet volumes from coarse model.")

    # Extract coarse schedule to use for warm-start
    coarse_schedule_for_1h_hotstart = extract_loader_schedule_list(decision_variables_2h)
    if not coarse_schedule_for_1h_hotstart:
         print("  Warning: Could not extract a valid schedule from the coarse model.")

    # Generate valid arcs to prune the fine model search space
    valid_combinations = generate_valid_arcs_for_all_loaders(
        decision_variables=decision_variables_2h,
        ratio=int(model_ver_coarse.get("t_p", 2) / model_ver_fine.get("t_p", 1)),
        buffer_pct=derived_2h.get("buffer_pct", 0.1)
    )
    print(f"  Generated {len(valid_combinations)} sets of valid arcs for pruning the fine model.")

    # Determine the final time horizon for the fine model
    max_tf_from_combinations = -1
    if valid_combinations:
        for loader_id in valid_combinations:
            if valid_combinations[loader_id]:
                max_tf_from_combinations = max(max_tf_from_combinations, max(valid_combinations[loader_id].keys()))

    T_length_from_combinations = max_tf_from_combinations + 1
    T_final_1h = max(T_heuristic_fine, T_length_from_combinations)
    print(f"  Determined fine model time horizon T = {T_final_1h}")

    #
    # === STEP 3: SOLVE FINE (1h) MODEL ===
    #
    print("\n4. Preparing Fine Model Data...")
    instance_raw_1h, derived_1h = prepare_loader_flow_data(
        instance_raw_fine,
        sens=sens_scn,
        model_ver=model_ver_fine,
        T_heuristic=T_final_1h,
        loader_routes_heuristic=None,  # Routes are now encoded in valid_arcs_at_time
        valid_arcs_at_time=valid_combinations,
        last_period_beet_volume=last_period_beet_volume
    )

    # Configure the specific warm-start mechanism
    coarse_hotstart_config_arg = None
    if time_agg_warmstart_params and coarse_schedule_for_1h_hotstart:
        ws_type = time_agg_warmstart_params.get('type')
        if ws_type:
            calculated_ratio = int(model_ver_coarse.get("t_p", 2) / model_ver_fine.get("t_p", 1))
            coarse_hotstart_config_arg = {**time_agg_warmstart_params, 'ratio': calculated_ratio}
            print(f"  Configured '{ws_type}' warm-start for fine model with ratio {calculated_ratio}.")

    print("5. Solving Fine Model...")

    scenario_results_1h = solve_loader_flow(
        instance=instance_raw_1h,
        derived=derived_1h,
        name=name,
        hotstart_solution=None, # Use the specialized coarse-to-fine hotstart below
        enforce_solution=None,
        vehicle_capacity_flag=vehicle_capacity_flag_fine,
        restricted_flow_flag=restricted_flow_flag_fine,
        last_period_restricted_flag=last_period_restricted_flag_fine,
        add_min_beet_restriction_flag=add_min_beet_restriction_flag_fine,
        coarse_hotstart_schedule=coarse_schedule_for_1h_hotstart,
        coarse_hotstart_config=coarse_hotstart_config_arg,
        FIXED_SOLVER_PARAMS=solver_params,
        verbose=verbose,
        combined_key=combined_key,
        base_file_path=base_file_path,
        inventory_flag=inventory_flag,
        inventory_cap_flag=inventory_cap_flag,
        v_type=v_type
    )

    print(f"\n--- Time Aggregation Finished for: {combined_key} ---\n")

    scenario_results = {
        "opt_type": "time_agg",
        "scenario_results_coarse": scenario_results_2h,
        "scenario_results_fine": scenario_results_1h
    }

    return scenario_results
