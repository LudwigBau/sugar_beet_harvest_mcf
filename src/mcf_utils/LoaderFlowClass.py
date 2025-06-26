import numpy as np
from gurobipy import GRB, Model, Var, LinExpr, quicksum  # Added Model, Var, LinExpr for Gurobi types
from typing import List, Tuple, Set, Dict, Optional, Any  # Added Any for generic Gurobi model
from collections import defaultdict

# Define Arc type alias
Arc = Tuple[int, int]  # Represents an arc from one location to another
LoaderKey = int  # Represents a loader ID
LocationKey = int  # Represents a location ID (field or depot)
TimeKey = int  # Represents a time period index

# Type alias for Gurobi decision variables dictionary
VarDict = Dict[Tuple, Var]


# --- Changes Start ---
# Loader Flow
class LoaderFlow:
    """
    Manages loader movement and operational states in a time-indexed formulation
    for harvest planning.

    This class defines decision variables for loader paths (x_l), working/idle
    states, and handles constraints related to flow conservation, operational
    rules (e.g., holidays), and travel times. It incorporates pruning
    strategies for variable creation based on pre-defined routes or
    time-specific valid arcs.
    """

    def __init__(self,
                 model: Model,
                 L: Set[LoaderKey],
                 Il: Dict[LoaderKey, List[LocationKey]],
                 I: Set[LocationKey],
                 c_l: Dict[LoaderKey, Any],  # Capacity or other loader-specific parameters
                 T: range,  # Set of time periods, typically range(Hf)
                 tau: Dict[LoaderKey, Any],  # Travel time data (e.g., pandas DataFrame tau[l].at[i,j])
                 access: Dict[LoaderKey, List[LocationKey]],
                 working_hours: Dict[LoaderKey, float],  # Max working hours per day for loader
                 holidays: Set[TimeKey],
                 v_type: str,  # Variable type: "binary" or "continuous"
                 time_period_length: float = 2.0,  # Duration of each time period in hours
                 loader_routes: Optional[Dict[LoaderKey, List[LocationKey]]] = None,
                 valid_arcs_at_time: Optional[Dict[LoaderKey, Dict[TimeKey, Set[Arc]]]] = None,
                 beet_volume: Optional[Dict[LocationKey, float]] = None,  # Work volume at each field
                 loader_rates: Optional[Dict[LoaderKey, float]] = None  # Processing rate for each loader
                 ):
        """
        Initializes the LoaderFlow instance.

        Args:
            model: The Gurobi model object.
            L: Set of loader IDs.
            Il: Dictionary mapping each loader ID to a list of accessible field/location IDs.
            I: Set of all field/location IDs (including depot, typically 0).
            c_l: Dictionary of loader capacities or other loader-specific parameters.
            T: Range object representing discrete time periods.
            tau: Travel time data structure (e.g., dict mapping loader to a matrix/df).
                 Expected to provide travel time via an '.at[from_loc, to_loc]' accessor
                 if using the default travel time logic.
            access: Dictionary mapping loader ID to a list of its accessible field IDs.
            working_hours: Dictionary mapping loader ID to max working hours per day.
            holidays: Set of time period indices that are holidays.
            v_type: Type of decision variables ("binary" or "continuous").
            time_period_length: Duration of a single time period in hours.
            loader_routes: Optional. Maps loader ID to a predefined static route (list of location IDs).
                           Used for pruning if `valid_arcs_at_time` is not provided.
            valid_arcs_at_time: Optional. Maps loader ID to time-indexed valid arcs.
                                Takes precedence over `loader_routes` for pruning.
                                Format: {loader: {time: {(from_loc, to_loc), ...}, ...}, ...}
            beet_volume: Optional. Maps field ID to the volume of work (e.g., beets to harvest).
                         Used for work-based route pruning.
            loader_rates: Optional. Also called L_bar. Maps loader ID to its processing rate (volume per hour).
                          Used for work-based route pruning.
        """
        self.model: Model = model
        self.L: Set[LoaderKey] = L
        self.I: Set[LocationKey] = I
        self.Il: Dict[LoaderKey, List[LocationKey]] = Il
        self.c_l: Dict[LoaderKey, Any] = c_l
        self.T: range = T
        self.tau: Dict[LoaderKey, Any] = tau  # Or more specific type if known, e.g. Dict[LoaderKey, pd.DataFrame]
        self.access: Dict[LoaderKey, List[LocationKey]] = access
        self.day_equivalent: int = int(np.round(24 / time_period_length))
        self.working_hours: Dict[LoaderKey, float] = working_hours
        self.holidays: Set[TimeKey] = holidays
        self.v_type: str = v_type
        self.time_period_length: float = time_period_length

        self.loader_routes: Optional[Dict[LoaderKey, List[LocationKey]]] = loader_routes
        self.valid_arcs_at_time: Optional[Dict[LoaderKey, Dict[TimeKey, Set[Arc]]]] = valid_arcs_at_time

        self.beet_volume: Dict[LocationKey, float] = beet_volume if beet_volume is not None else {}
        self.loader_rates: Dict[LoaderKey, float] = loader_rates if loader_rates is not None else {}

        # Decision variables (Gurobi Var objects, typically stored in dicts)
        self.x: Optional[VarDict] = None  # x[l,i,j,t]: loader l moves i to j at time t
        self.working: Optional[VarDict] = None  # working[l,i,t]: loader l works at i at time t
        self.idle: Optional[VarDict] = None  # idle[l,i,t]: loader l is idle at i at time t
        self.y: Optional[VarDict] = None  # Placeholder for y variables if used elsewhere
        self.y_in: Optional[VarDict] = None  # y_in[l,i,j,t]: travel time related var
        self.y_out: Optional[VarDict] = None  # y_out[l,i,j,t]: travel time related var
        self.started: Optional[VarDict] = None  # started[l,i,t]: operation started
        self.penalty: Optional[VarDict] = None  # Placeholder for penalty variables
        self.finished: Optional[VarDict] = None  # finished[l,i,t]: operation finished
        self.restricted: Optional[VarDict] = None  # restricted[l,i,t]: loader is in restricted state

        # Sets for quick lookups of valid stay locations based on created x variables
        self.valid_stay_lit: Set[Tuple[LoaderKey, LocationKey, TimeKey]] = set()
        self.valid_stay_lit_no_depot: Set[Tuple[LoaderKey, LocationKey, TimeKey]] = set()
        self.last_period_combinations = set()

        self.create_decision_variables(v_type)

    def create_decision_variables(self, v_type):
        """
        Create valid combinations (l, i, j, t) for the decision variables x.
        Prioritizes pruning based on `valid_arcs_at_time`.
        If not provided, falls back to `loader_routes`.
        If loader_routes is used, applies an additional time-based pruning
        if beet_volume and loader_rates are available.
        If neither is provided, uses the naive full combination.
        """
        valid_combinations = []
        print(f"Creating loader decision variables (type: {v_type})...")

        if self.valid_arcs_at_time is not None:
            # ... (your existing code for valid_arcs_at_time) ...
            print("Using time-dependent valid arc sets for pruning.")
            # --- Pruning based on time-dependent valid arcs ---
            for l in self.L:
                if l not in self.valid_arcs_at_time:
                    print(
                        f"Warning: valid_arcs_at_time provided, but missing entry for loader {l}. "
                        f"Using full set for this loader based on Il[l].")  # Fallback for this loader
                    for i in self.Il[l]:
                        for j in self.Il[l]:
                            for t in self.T:
                                valid_combinations.append((l, i, j, t))
                    continue

                loader_valid_arcs = self.valid_arcs_at_time[l]
                for t in self.T:
                    if t in loader_valid_arcs:
                        for arc in loader_valid_arcs[t]:
                            i, j_node = arc  # Renamed j to j_node to avoid conflict
                            if i in self.Il[l] and j_node in self.Il[l]:
                                valid_combinations.append((l, i, j_node, t))
                            else:
                                print(f"Warning: Arc ({i},{j_node}) at time {t} for loader {l} involves locations "
                                      f"not in Il[{l}]. Skipping.")
                    # Ensure staying at the depot (0,0) is always possible if 0 is in Il[l]
                    # This might be redundant if valid_arcs_at_time is comprehensive
                    # but can be a safeguard.
                if 0 in self.Il[l]:  # Ensure depot stay is considered
                    for t_period in self.T:  # Renamed t to t_period
                        # Check if (l,0,0,t_period) is already added by valid_arcs_at_time[l][t_period]
                        # To avoid duplicates if user provides it.
                        # However, set operations later will handle duplicates.
                        valid_combinations.append((l, 0, 0, t_period))

        elif self.loader_routes is not None:

            print("Using static loader routes for pruning.")

            loader_earliest_start_time_at_node = {}  # Dict[loader_id, Dict[node_id, min_time_period]]
            can_use_work_based_pruning = bool(self.beet_volume and self.loader_rates)
            if can_use_work_based_pruning:
                print("Applying additional work-based time pruning for routes.")

                for l_calc in self.L:
                    if l_calc not in self.loader_routes or l_calc not in self.loader_rates or self.loader_rates[
                        l_calc] <= 0:
                        continue  # Skip if no route, no rate, or invalid rate for this loader

                    route = self.loader_routes[l_calc]
                    current_earliest_finish_time_periods = 0
                    est_for_loader = {}  # Dict[node_id, min_time_period]

                    for node_on_route in route:

                        # If node_on_route is encountered for the first time, record its earliest start time.
                        # Do not update if it already exists (e.g. depot visited at start and end of route).
                        if node_on_route not in est_for_loader:
                            est_for_loader[node_on_route] = current_earliest_finish_time_periods

                        # The print statement now reflects the EST that was set (or already existed)
                        # print( f"Loader:
                        # {l_calc}, pos. {node_on_route}, EST set/kept: {est_for_loader.get(node_on_route)},
                        # " f"current earliest finish time before this node's work: {
                        # current_earliest_finish_time_periods}")
                        work_periods_at_node = 0
                        # Assuming depot is 0 and has no work in this calculation context
                        if node_on_route != 0 and node_on_route in self.beet_volume:
                            # Ensure loader_rates[l_calc] is not zero to avoid DivisionByZeroError,
                            # already handled by the check at the start of the l_calc loop.
                            work_hours_at_node = self.beet_volume[node_on_route] / self.loader_rates[l_calc]
                            work_periods_at_node = np.ceil(work_hours_at_node / self.time_period_length).astype(int)

                        current_earliest_finish_time_periods += work_periods_at_node
                    loader_earliest_start_time_at_node[l_calc] = est_for_loader
                # print("loader_earliest_start_time_at_node: ", loader_earliest_start_time_at_node)  # Adjusted print key
            else:
                print("Skipping work-based time pruning: beet_volume or loader_rates not fully provided.")

            # --- Pruning based on static routes (and potentially work-based time pruning) ---
            for l in self.L:
                if l not in self.loader_routes:
                    # If using routes, expect all loaders intended for this logic to have one.
                    # Or handle this loader with a different strategy (e.g. naive full if Il[l] is its scope)
                    print(
                        f"Warning: Static loader_routes provided, but route missing for loader {l}. "
                        f"Skipping variable creation for this loader under this rule.")
                    continue  # Or raise ValueError as in original code if routes are mandatory for all L

                route = self.loader_routes[l]
                min_start_times_for_l_nodes = loader_earliest_start_time_at_node.get(l, {})  # Empty dict if no profile

                route_arcs = set()
                # Add arcs along the route
                for idx in range(len(route) - 1):
                    i_node = route[idx]
                    j_node = route[idx + 1]
                    if i_node in self.Il[l] and j_node in self.Il[l]:
                        route_arcs.add((i_node, j_node))
                    else:
                        print(
                            f"Warning: Route for loader {l} contains invalid step ({i_node} -> {j_node}). "
                            f"Skipping this step.")

                # Add arcs from route nodes back to depot (if 0 is a valid location for loader)
                if 0 in self.Il[l]:
                    for node_in_route in route:
                        if node_in_route in self.Il[l]:  # Ensure node itself is valid for loader
                            route_arcs.add((node_in_route, 0))

                # Add stay arcs for all nodes in the route and the depot
                for node_in_route in route:
                    if node_in_route in self.Il[l]:
                        route_arcs.add((node_in_route, node_in_route))
                if 0 in self.Il[l]:  # Ensure depot stay is possible
                    route_arcs.add((0, 0))

                # Create combinations for all times for these allowed arcs
                for i_arc, j_arc in route_arcs:  # Renamed i,j to avoid conflict
                    # Min time to START an arc FROM i_arc
                    min_t_for_arc_departure = min_start_times_for_l_nodes.get(i_arc, 0)

                    for t_period in self.T:  # Renamed t to t_period
                        if t_period >= min_t_for_arc_departure:
                            valid_combinations.append((l, i_arc, j_arc, t_period))
                        # else:  # For debugging
                        #    print(f"Pruning x({l},{i_arc},{j_arc},{t_period}) due to "
                        #          f"EST[{i_arc}]={min_t_for_arc_departure}")


        else:
            print("Using naive full combinations (no pruning).")
            for l in self.L:
                for i in self.Il[l]:
                    for j_node in self.Il[l]:  # Renamed j to j_node
                        for t_period in self.T:  # Renamed t to t_period
                            valid_combinations.append((l, i, j_node, t_period))

        # Ensure unique combinations
        unique_combinations = sorted(list(set(valid_combinations)))  # Sorting helps in debugging/consistency
        valid_combinations = unique_combinations  # Use the unique, sorted list

        if not valid_combinations:
            print("Warning: No valid (l, i, j, t) combinations were generated. Check pruning logic and inputs.")
            self.x = {}
            self.valid_stay_lit = set()  # Initialize to empty
            self.valid_stay_lit_no_depot = set()  # Initialize to empty
            return

        print(f"Created {len(valid_combinations)} unique (l, i, j, t) combinations for x_l variables.")
        # print("Valid Combinations:", valid_combinations)

        # Create the decision variables x[l,i,j,t]
        if v_type == "binary":
            self.x = self.model.addVars(valid_combinations,
                                        vtype=GRB.BINARY,
                                        name="x_l")
        elif v_type == "continuous":
            self.x = self.model.addVars(valid_combinations,
                                        vtype=GRB.CONTINUOUS,
                                        lb=0,
                                        ub=1,  # Keep ub=1 for continuous relaxation
                                        name="x_l")
        else:
            raise ValueError(f"Unsupported variable type: {v_type}. Choose 'binary' or 'continuous'.")

        # --- CHANGE: Populate valid stay sets based on created x variables ---
        # Identify (l, i, t) where a stay arc x[l,i,i,t] exists
        self.valid_stay_lit = set()
        for key in self.x.keys():  # Iterate through the keys of the created x variables
            l, i, j, t = key
            if i == j:  # Check if it's a stay arc
                self.valid_stay_lit.add((l, i, t))

        # Create the subset excluding the depot (i=0)
        self.valid_stay_lit_no_depot = set(
            (l, i, t) for l, i, t in self.valid_stay_lit if i != 0
        )
        print(f"Identified {len(self.valid_stay_lit)} valid (l, i, t) stay combinations.")
        print(f"Identified {len(self.valid_stay_lit_no_depot)} valid non-depot (l, i, t) stay combinations.")
        # --- End CHANGE ---

        self.model.update()  # Update model to make variables available

    def add_core_constraints(self):
        """
        Add the main model constraints using the valid arc set.
        These replicate the classic time-indexed formulation:
         - Flow conservation over time.
         - Exactly one arc is chosen at each time period.
         - Initial condition: start at depot.
        """
        # Flow conservation: for each loader l, each available location i and for each t >= 1.
        for l in self.L:
            for i in self.Il[l]:
                for t in self.T[1:]:
                    # Consider only valid arcs from the pruned set:
                    out_arcs = [(l, i, j, t) for j in self.Il[l] if (l, i, j, t) in self.x]
                    in_arcs = [(l, k, i, t - 1) for k in self.Il[l] if (l, k, i, t - 1) in self.x]
                    self.model.addConstr(
                        quicksum(self.x[arc] for arc in out_arcs) ==
                        quicksum(self.x[arc] for arc in in_arcs),
                        f"flow_conservation_l_{l}_i_{i}_t_{t}"
                    )

        # One place at a time: at each time period, exactly one valid arc is chosen.
        for l in self.L:
            for t in self.T:
                valid_arcs = [(l, i, j, t) for i in self.Il[l] for j in self.Il[l] if (l, i, j, t) in self.x]
                self.model.addConstr(
                    quicksum(self.x[arc] for arc in valid_arcs) == 1,
                    f"one_place_at_a_time_l_{l}_t_{t}"
                )

        # Initial condition: At time t=0, force the loader to be at the depot.
        for l in self.L:
            for i in self.Il[l]:
                for j in self.Il[l]:
                    key = (l, i, j, 0)
                    if key in self.x:
                        if i == 0 and j == 0:
                            self.model.addConstr(self.x[l, i, j, 0] == 1,
                                                 f"init_l_{l}_i_{i}_j_{j}_0")
                        else:
                            self.model.addConstr(self.x[l, i, j, 0] == 0,
                                                 f"init_l_{l}_i_{i}_j_{j}_0")

        # Outgoing Movement Constraint: For each loader l and each location i,
        # ensure that the sum of all x[l, i, j, t] (over t and over j with i != j)
        # is at most 1.
        for l in self.L:
            for i in self.Il[l]:
                valid_out_arcs = [(l, i, j, t) for t in self.T for j in self.Il[l] if
                                  i != j and (l, i, j, t) in self.x]
                self.model.addConstr(
                    quicksum(self.x[arc] for arc in valid_out_arcs) <= 1,
                    name=f"restrict_loader_outgoing_{l}_{i}"
                )

        # Incoming Movement Constraint: For each loader l and each location j,
        # ensure that the sum of all x[l, k, j, t] (over t and over k with k != j)
        # is at most 1.
        for l in self.L:
            for j in self.Il[l]:
                valid_in_arcs = [(l, k, j, t) for t in self.T for k in self.Il[l] if
                                 k != j and (l, k, j, t) in self.x]
                self.model.addConstr(
                    quicksum(self.x[arc] for arc in valid_in_arcs) <= 1,
                    name=f"restrict_loader_incoming_{l}_{j}"
                )

    def add_idle(self, v_type):
        """ Create idle/working variables only for valid stay locations/times. """
        # --- CHANGE: Use self.valid_stay_lit for combinations ---
        idle_working_combinations = list(self.valid_stay_lit)

        if not idle_working_combinations:
            print("Warning: No valid stay combinations found, cannot create idle/working variables.")
            self.idle = {}
            self.working = {}
            return

        print(f"Creating {len(idle_working_combinations)} idle/working variables...")
        # Decision variables
        var_type = GRB.BINARY if v_type == "binary" else GRB.CONTINUOUS
        self.idle = self.model.addVars(idle_working_combinations,
                                       vtype=var_type, lb=0, ub=1, name="idle_l")
        self.working = self.model.addVars(idle_working_combinations,
                                          vtype=var_type, lb=0, ub=1, name="working_l")
        # --- End CHANGE ---

        # Split X (only for existing stay arcs)
        # --- CHANGE: Iterate over valid combinations ---
        for (l, i, t) in idle_working_combinations:
            # The key (l,i,i,t) for x is guaranteed tyo exist because
            # idle_working_combinations was derived from existing stay arcs in x
            key_x = (l, i, i, t)
            if key_x not in self.x:
                # This check is technically redundant now but safe to keep
                print(f"Warning: x key {key_x} not found unexpectedly in add_idle. Skipping constraint.")
                continue
            # Ensure idle and working variables were created for this combo
            key_idle_work = (l, i, t)
            if key_idle_work not in self.idle or key_idle_work not in self.working:
                print(f"Warning: idle/working key {key_idle_work} not found in add_idle. Skipping constraint.")
                continue

            self.model.addConstr(self.x[key_x] == self.idle[key_idle_work] + self.working[key_idle_work],
                                 name=f"idle_working_constraint_{l}_{i}_{t}")
        # --- End CHANGE ---
        self.model.update()

    def add_hard_restricted(self):
        """ Create restricted-related variables only for valid non-depot stay locations/times. """
        # --- CHANGE: Use self.valid_stay_lit_no_depot for combinations ---
        restricted_combinations = list(self.valid_stay_lit_no_depot)

        if not restricted_combinations:
            print("Warning: No valid non-depot stay combinations found, cannot create restricted variables.")
            self.started = {}
            self.finished = {}
            self.restricted = {}
            return

        print(f"Creating {len(restricted_combinations)} started/finished/restricted variables over available t...")
        # Add new decision variables for start, finished and restricted
        self.started = self.model.addVars(restricted_combinations, vtype=GRB.BINARY, lb=0, ub=1, name="started_l")
        self.finished = self.model.addVars(restricted_combinations, vtype=GRB.BINARY, lb=0, ub=1, name="finished_l")
        self.restricted = self.model.addVars(restricted_combinations, vtype=GRB.BINARY, lb=0, ub=1, name="restricted_l")
        # --- End CHANGE ---

        # Constraints related to restricted variables
        # --- CHANGE: Iterate over valid combinations ---
        for (l, i, t) in restricted_combinations:  # i is guaranteed != 0 here
            key_restr = (l, i, t)
            # Ensure vars exist (technically redundant but safe)
            if key_restr not in self.started or key_restr not in self.finished or key_restr not in self.restricted:
                print(f"Warning: restricted variables key {key_restr} "
                      f"not found in add_hard_restricted. Skipping constraint.")
                continue
            self.model.addConstr(self.restricted[key_restr] == self.started[key_restr] - self.finished[key_restr],
                                 f"restr_def_{l}_{i}_{t}")  # Changed name slightly

        # restricted <= 1 constraint needs careful handling with quicksum
        # Group restricted vars by (l, t)
        vars_by_lt = defaultdict(list)
        for (l, i, t) in restricted_combinations:
            vars_by_lt[(l, t)].append(self.restricted[l, i, t])

        for (l, t), vars_list in vars_by_lt.items():
            if t > 0 and vars_list:  # Ensure t > 0 and list is not empty
                self.model.addConstr(quicksum(vars_list) <= 1,
                                     f"restricted_le_1_l{l}_t{t}")  # Changed name slightly
        # --- End CHANGE ---
        self.model.update()

    def add_last_period_restricted(self):

        """ Create restricted-related variables only for valid non-depot stay locations at last period. """
        self.last_period_combinations = set()
        for l in self.L:
            for i in self.Il[l]:
                if i != 0:
                    t = max(self.T)
                    self.last_period_combinations.add((l, i, t))

        print("Created last_period_combinations: \n", self.last_period_combinations)
        restricted_combinations = list(self.last_period_combinations)

        if not restricted_combinations:
            print("Warning: No valid non-depot stay combinations found, cannot create restricted variables.")
            self.started = {}
            self.finished = {}
            self.restricted = {}
            return

        print(f"Creating {len(restricted_combinations)} started/finished/restricted variables...")
        # Add new decision variables for start, finished and restricted
        self.started = self.model.addVars(restricted_combinations, vtype=GRB.BINARY, lb=0, ub=1, name="started_l")
        self.finished = self.model.addVars(restricted_combinations, vtype=GRB.BINARY, lb=0, ub=1, name="finished_l")
        self.restricted = self.model.addVars(restricted_combinations, vtype=GRB.BINARY, lb=0, ub=1, name="restricted_l")
        # --- End CHANGE ---

        # Constraints related to restricted variables
        # --- CHANGE: Iterate over valid combinations ---
        for (l, i, t) in restricted_combinations:  # i is guaranteed != 0 here
            key_restr = (l, i, t)
            # Ensure vars exist (technically redundant but safe)
            if key_restr not in self.started or key_restr not in self.finished or key_restr not in self.restricted:
                print(f"Warning: restricted variables key {key_restr} "
                      f"not found in add_hard_restricted. Skipping constraint.")
                continue
            self.model.addConstr(self.restricted[key_restr] == self.started[key_restr] - self.finished[key_restr],
                                 f"restr_def_{l}_{i}_{t}")  # Changed name slightly

        # restricted <= 1 constraint needs careful handling with quicksum
        for l in self.L:
            self.model.addConstr(quicksum(self.restricted[l, i, max(self.T)] for i in self.Il[l] if i != 0) <= 1,
                                 f"restricted_le_1_l{l}_t{max(self.T)}")
        # --- End CHANGE ---
        self.model.update()

    def add_simple_travel_time(self):
        """
        Create partial travel time variables and add constraints using the pruned valid combination set
        from self.x. In this formulation, we define y_in and y_out only on those arcs (l, i, j, t)
        that were created in the valid combination process (i.e. they appear in self.x.keys()).

        For each arc (l, i, j, t) in self.x:
          - If i != j (travel arc):
               y_in[l,i,j,t] + y_out[l,i,j,t] <= (1 - tau[l].at[i,j]) * x[l,i,j,t].
          - If i == j (stay arc):
               y_in[l,i,i,t] + y_out[l,i,i,t] == working[l,i,t]
           and enforce:
               y_in[l,i,i,t] == y_out[l,i,i,t].
        """
        # Define the valid domain from the pruned x-variables.
        valid_y = list(self.x.keys())

        # Create y_in and y_out only for these valid arcs.
        self.y_in = self.model.addVars(valid_y,
                                       vtype=GRB.CONTINUOUS,
                                       lb=0,
                                       ub=1,
                                       name="y_in")
        self.y_out = self.model.addVars(valid_y,
                                        vtype=GRB.CONTINUOUS,
                                        lb=0,
                                        ub=1,
                                        name="y_out")

        # Loop over each valid arc and add the appropriate constraint.
        for (l, i, j, t) in valid_y:
            if i != j:
                self.model.addConstr(
                    self.y_in[l, i, j, t] + self.y_out[l, i, j, t] <= (1 - self.tau[l].at[i, j]) * self.x[l, i, j, t],
                    name=f"partial_period_ij_{l}_{i}_{j}_{t}"
                )
            else:
                # For "stay" arcs, tie the y variables to the working variable.
                self.model.addConstr(
                    self.y_in[l, i, i, t] + self.y_out[l, i, i, t] == self.working[l, i, t],
                    name=f"partial_period_stay_{l}_{i}_{t}"
                )
                self.model.addConstr(
                    self.y_in[l, i, i, t] == self.y_out[l, i, i, t],
                    name=f"partial_balance_stay_{l}_{i}_{t}"
                )
        self.model.update()

    def add_holiday_constraint(self):
        """
        Prevents loaders from working at non-depot locations or initiating new travel
        during holiday periods. Constraints are applied only if the corresponding
        decision variables (working, x) exist for the given (loader, location(s), time) combination.
        """
        print("Holidays in LoaderClass", self.holidays)
        if not self.holidays:
            # print("No holidays defined, skipping holiday constraints.") # Optional: for debugging
            return

        for l in self.L:
            for t_holiday in self.holidays:  # Iterate through each defined holiday period
                if t_holiday not in self.T:  # Ensure the holiday falls within the model's time horizon
                    continue

                # Iterate through all locations 'i' available to loader 'l'
                # 'i' will serve as the current location for 'working' checks,
                # and as the origin for 'x' (travel) checks.
                for i in self.Il[l]:
                    # Constraint 1: Prevent 'working' at non-depot locations on holidays.
                    # 'working' variables are defined for (l, i, t) where a loader 'l' stays at 'i' at 't'.
                    if i != 0:  # Non-depot locations
                        key_working = (l, i, t_holiday)
                        if key_working in self.working:  # Check if the working variable exists
                            self.working[key_working].ub = 0

                    # Constraint 2: Prevent initiating travel (x[l,i,j,t] where i != j) on holidays.
                    # Loaders should not start a new trip between different locations.
                    # They can remain where they are (x[l,i,i,t_holiday] = 1 is allowed,
                    # but working[l,i,t_holiday] would be 0 if i != 0).
                    # Iterate through all possible destination locations 'j' for loader 'l'.
                    for j in self.Il[l]:
                        if i != j:  # Only consider travel arcs (origin 'i' to different destination 'j')
                            key_x = (l, i, j, t_holiday)
                            if key_x in self.x:  # Check if the travel variable x[l,i,j,t] exists
                                self.x[key_x].ub = 0

        self.model.update()  # Assuming model.update() is called later, or if specific to this class, can be added.
