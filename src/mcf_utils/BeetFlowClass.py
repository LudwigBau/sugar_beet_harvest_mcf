import pandas as pd
import numpy as np
from gurobipy import GRB, quicksum

# Define these at the class level or ensure they are accessible if used in multiple methods
# States: 0 = Field Heap, 1 = Storage, 2 = Production
ALLOWED_BEET_FLOW_TRANSITIONS = [
    (0, 0),  # Stay in Field Heap
    (0, 1),  # Field Heap to Storage
    (0, 2),  # Field Heap to Production
    (1, 1),  # Stay in Storage
    (1, 2),  # Storage to Production
    (2, 2)  # Stay in Production
]

class BeetFlow:
    """
    This class sets up beet flow related constraints and functions.
    """

    def __init__(self, model, I, T, beet_volume, inventory=None, last_period_beet_volume=None):
        self.model = model
        self.I = I  # Fields (Ih without depot)
        self.T = T
        self.volume = beet_volume
        self.b = None  # Decision variables for beet flow
        self.inventory_level = max(inventory if inventory is not None else 0, 0)
        self.last_period_beet_volume = last_period_beet_volume

    def add_load_only_flow_conservation_constraint(self):
        """
        Adds flow conservation constraints for beet flow (f,s,p),
        creating variables only for allowed unidirectional flows.
        Allowed flows: 0->0, 0->1, 0->2, 1->1, 1->2, 2->2.
        No backward flows like 1->0, 2->0, or 2->1.
        """
        # --- CHANGES START ---

        # Create a list of keys for only allowed transitions
        b_variable_keys = [
            (i, from_state, to_state, t)
            for i in self.I
            for from_state, to_state in ALLOWED_BEET_FLOW_TRANSITIONS
            for t in self.T
        ]

        # Create Decision Variables only for the allowed keys
        self.b = self.model.addVars(b_variable_keys, vtype=GRB.CONTINUOUS, lb=0, name="b")

        # --- CHANGES END ---

        # Flow conservation for beets. From field (f), to (store in) inventory (s), to production (p)
        # Direct flow from harvest to production is also allowed
        # These constraints inherently use only the allowed flow variables based on their structure.
        self.model.addConstrs(
            (self.b[i, 0, 0, t] == self.b[i, 0, 0, t - 1] - self.b[i, 0, 1, t] - self.b[i, 0, 2, t]
             for i in self.I for t in self.T if t > 0), "b_flow_ff")
        self.model.addConstrs(
            (self.b[i, 1, 1, t] == self.b[i, 1, 1, t - 1] + self.b[i, 0, 1, t] - self.b[i, 1, 2, t] for i in self.I for
             t in self.T if t > 0), "b_flow_ss")
        self.model.addConstrs(
            (self.b[i, 2, 2, t] == self.b[i, 2, 2, t - 1] + self.b[i, 1, 2, t] + self.b[i, 0, 2, t] for i in self.I for
             t in self.T if t > 0), "b_flow_pp")

        if self.inventory_level is not None:  # inventory_level is already max(..., 0)
            print("Set up initial inventory levels in BeetFlowClass:", self.inventory_level)

        for i in self.I:
            if isinstance(self.volume, pd.DataFrame):
                init_value = self.volume.at[i, "0"]
            elif isinstance(self.volume, np.ndarray):
                init_value = self.volume[i - 1]
            elif isinstance(self.volume, dict):
                init_value = self.volume[i]
            else:
                raise TypeError("Unsupported volume type or shape")

            # Set init values for state variables (these keys will exist as per ALLOWED_BEET_FLOW_TRANSITIONS)
            self.model.addConstr(self.b[i, 0, 0, 0] == init_value, f"init_b_{i}_0_0_0")
            self.model.addConstr(self.b[i, 2, 2, 0] == 0, f"init_b_{i}_2_2_0")

            # Determine the upper bound value for variables related to field i
            # This UB value will be applied to all existing b[i,from,to,t] variables for this field i.
            current_ub_for_field_i_variables = init_value
            if self.inventory_level is not None:  # Already known to be >= 0 if not None
                current_ub_for_field_i_variables += (self.inventory_level / len(self.I))

            # Set Inventory initial condition
            if self.inventory_level is not None:
                self.model.addConstr(self.b[i, 1, 1, 0] == (self.inventory_level / len(self.I)),
                                     f"init_b_{i}_1_1_0")
            else:  # No inventory_level provided (or it was 0 initially)
                self.model.addConstr(self.b[i, 1, 1, 0] == 0, f"init_b_{i}_1_1_0")

            # Set upper bounds only for the created variables associated with field i
            for t_loop in self.T:
                for from_state, to_state in ALLOWED_BEET_FLOW_TRANSITIONS:
                    key = (i, from_state, to_state, t_loop)
                    # self.b was created using b_variable_keys, so self.b[key] will exist.
                    self.b[key].UB = current_ub_for_field_i_variables

    def add_full_flow_conservation_constraint(self):
        """
        Adds flow conservation constraints for beet flow (f,h,i,p).
        """
        # Create Decision Variable
        self.b = self.model.addVars(self.I, 4, 4, self.T, vtype=GRB.CONTINUOUS, lb=0, name="b")

        # Flow conservation for beets. From field (f), to harvested (h), to (store in) inventory (s), to production (p)
        # Direct flow from harvest to production is also allowed
        self.model.addConstrs(
            (self.b[i, 0, 0, t] == self.b[i, 0, 0, t - 1] - self.b[i, 0, 1, t] for i in self.I for t in self.T if
             t > 0), "b_flow_ff")
        self.model.addConstrs(
            (self.b[i, 1, 1, t] == self.b[i, 1, 1, t - 1] + self.b[i, 0, 1, t] - self.b[i, 1, 2, t] - self.b[i, 1, 3, t]
             for i in self.I for t in self.T if t > 0), "b_flow_hh")
        self.model.addConstrs(
            (self.b[i, 2, 2, t] == self.b[i, 2, 2, t - 1] + self.b[i, 1, 2, t] - self.b[i, 2, 3, t] for i in self.I for
             t in self.T if t > 0), "b_flow_ss")
        self.model.addConstrs(
            (self.b[i, 3, 3, t] == self.b[i, 3, 3, t - 1] + self.b[i, 1, 3, t] + self.b[i, 2, 3, t] for i in self.I for
             t in self.T if t > 0), "b_flow_pp")

        # Assuming self.model and self.b are already defined in your context
        for i in self.I:
            if isinstance(self.volume, pd.DataFrame):
                init_value = self.volume.at[i, "0"]
            elif isinstance(self.volume, np.ndarray):  # For numpy arrays
                init_value = self.volume[i - 1]  # Adjust for 0-based indexing in numpy
            elif isinstance(self.volume, dict):  # Correct for dictionary case
                init_value = self.volume[i]  # Access value by key in the dictionary
            else:
                raise TypeError("Unsupported volume type or shape")

            self.model.addConstr(self.b[i, 0, 0, 0] == init_value, f"init_b_{i}_0_0_0")
            self.model.addConstr(self.b[i, 1, 1, 0] == 0, f"init_b_{i}_1_1_0")
            self.model.addConstr(self.b[i, 2, 2, 0] == 0, f"init_b_{i}_2_2_0")
            self.model.addConstr(self.b[i, 3, 3, 0] == 0, f"init_b_{i}_3_3_0")

    def add_min_beet_restriction(self):
        print("Add add_min_beet_restriction() <=> loading goals")

        # Check if we got values
        if not self.last_period_beet_volume:
            raise ValueError("last_period_beet_volume is empty in BeetFlowClass. Prepare data accordingly")

        # Check if all keys of the dict last_period_beet_volume are in the set I
        if not all(key in self.I for key in self.last_period_beet_volume):
            raise KeyError("Not all keys in last_period_beet_volume are in self.I in BeetFlowClass")

        print("Last period loading goals: \n", self.last_period_beet_volume)
        # Set loading goals accordingly
        for i in self.I:
            self.model.addConstr(
                self.b[i, 0, 0, max(self.T)] <= self.last_period_beet_volume[i],
                f"beet_goal{i}")
