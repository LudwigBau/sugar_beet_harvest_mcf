import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

# TODO: Fix bug

"""
Debugging: 

3.  Transfer Inventory levels minus daily production volume to nect week!
   
4.  Insert Holiday Constraint and mechanisms into gurobi model? (or not and argue why holidays are left our, argue
    that mechanism work for sunday and thus for all other holidays, it is more of a production plan optimisation than
    model)
"""


class ProductionPlanning:
    def __init__(self, total_volume, production_volume_per_day, machine_ids, assigned_volume_per_machine,
                 verbose=False):

        self.total_volume = total_volume
        self.production_volume_per_day = production_volume_per_day
        self.assigned_volume_per_machine = list(assigned_volume_per_machine.values())

        # Validate that the total volume matches the sum of assigned volumes
        if sum(self.assigned_volume_per_machine) != self.total_volume:
            print("Sum assigned volume: ", self.assigned_volume_per_machine)
            print("Total Volume ", self.total_volume)
            raise ValueError("The sum of assigned volumes per machine must equal the total volume.")

        self.machine_names = machine_ids

        self.daily_goal_per_machine = None
        self.days_to_complete = None
        self.df = None

        self.verbose = verbose

    def calculate_production_plan(self):
        if self.verbose:
            print("Calculating production plan...")

        # Calculate the proportion of volume each machine is responsible for
        volume_ratios = np.array(self.assigned_volume_per_machine) / self.total_volume

        if self.verbose:
            print("Production Volume per Day:", self.production_volume_per_day)
            print("Machine Ratios:", volume_ratios)
        # Calculate the daily goal per machine based on their assigned volume ratio
        self.daily_goal_per_machine = volume_ratios * self.production_volume_per_day

        # Calculate the number of days required for each machine to complete its assigned volume
        self.days_to_complete = np.ceil(
            np.array(self.assigned_volume_per_machine) / self.daily_goal_per_machine).astype(int)

        if self.verbose:
            print("Daily goals per machine:", self.daily_goal_per_machine)
            print("Days to complete per machine:", self.days_to_complete)

    def redistribute_holiday_demand(self, start_date, holidays):
        if self.daily_goal_per_machine is None or self.days_to_complete is None:
            raise ValueError("You must calculate the production plan before redistributing demand.")

        # Calculate the maximum number of days required based on the longest time needed by any machine
        num_days = max(self.days_to_complete)

        if self.verbose:
            print(f"Redistributing holiday demand for {num_days} days, starting from {start_date}...")

        # Create a date range starting from the start_date
        self.date_range = pd.date_range(start=start_date, periods=num_days, freq='D')

        # Normalize holidays to dates without time components
        holidays_normalized = pd.to_datetime(holidays).normalize()
        self.holiday_dates = set(holidays_normalized)

        # Initialize holiday flags
        self.holiday_flags = np.zeros(len(self.date_range), dtype=bool)

        # Identify Sundays and holidays
        for i, date in enumerate(self.date_range):
            date_normalized = date.normalize()
            if date.weekday() == 6 or date_normalized in self.holiday_dates:  # Sunday or holiday
                self.holiday_flags[i] = True

        # Initialize daily_goals with initial per-day per-machine demands
        num_machines = len(self.machine_names)
        self.daily_goals = np.zeros((len(self.date_range), num_machines))

        # For each machine, calculate initial per-day demands
        for machine_index in range(num_machines):
            remaining_volume = self.assigned_volume_per_machine[machine_index]
            daily_goal = self.daily_goal_per_machine[machine_index]
            for day in range(len(self.date_range)):
                if remaining_volume <= 0:
                    self.daily_goals[day, machine_index] = 0
                else:
                    if remaining_volume < daily_goal:
                        daily_production = remaining_volume
                    else:
                        daily_production = daily_goal
                    self.daily_goals[day, machine_index] = daily_production
                    remaining_volume -= daily_production

        # Save original daily goals before redistribution
        original_daily_goals = self.daily_goals.copy()

        # Set demands to zero on holidays
        self.daily_goals[self.holiday_flags, :] = 0

        # Process each holiday individually, allowing overlapping redistributions
        holiday_indices = np.where(self.holiday_flags)[0]

        for h_idx in holiday_indices:
            # if self.verbose:
            # print(f"\nRedistributing demand for holiday on {self.date_range[h_idx].date()}...")
            # For each machine, get the demand that was planned on the holiday
            holiday_demand_per_machine = original_daily_goals[h_idx, :]
            # If there's no demand on the holiday, skip redistribution
            if np.all(holiday_demand_per_machine == 0):
                if self.verbose:
                    print("No demand on holiday, skipping redistribution.")
                continue

            # Find prior 6 workdays (excluding holidays and Sundays)
            prior_workdays = []
            idx = h_idx - 1
            while len(prior_workdays) < 6 and idx >= 0:
                if not self.holiday_flags[idx]:
                    prior_workdays.append(idx)
                idx -= 1

            if len(prior_workdays) < 6:
                if self.verbose:
                    print(f"Cannot redistribute demand due to insufficient prior workdays.")
                continue  # Cannot redistribute, insufficient prior workdays

            prior_workdays = prior_workdays[::-1]  # Reverse to chronological order

            # if self.verbose:
            # print(f"Redistributing to prior workdays: {[self.date_range[i].date() for i in prior_workdays]}")

            # Proceed to redistribute the demand
            for machine_index in range(num_machines):
                holiday_demand = holiday_demand_per_machine[machine_index]
                increment = holiday_demand / 6

                # Add the increment to each of the prior 6 workdays
                for idx in prior_workdays:
                    self.daily_goals[idx, machine_index] += increment
                    # if self.verbose:
                    #    print(f"Machine {self.machine_names[machine_index]}: Added {increment} units to {self.date_range[idx].date()}")

    def adjust_minimum_daily_demand(self, min_demand=1000):
        if self.daily_goals is None:
            raise ValueError("You must redistribute holiday demand before adjusting minimum daily demand.")

        if self.verbose:
            print("\nAdjusting daily goals to ensure minimum daily volume per machine per week...")

        num_machines = len(self.machine_names)
        week_numbers = pd.Series(self.date_range).dt.isocalendar().week.values

        for machine_index in range(num_machines):
            # if self.verbose:
            #    print(f"\nAdjusting Machine {self.machine_names[machine_index]}...")
            for week_number in np.unique(week_numbers):
                # Get indices of dates in this week
                indices = np.where(week_numbers == week_number)[0]

                # Exclude holidays
                working_indices = indices[~self.holiday_flags[indices]]

                if len(working_indices) == 0:
                    if self.verbose:
                        print(f"Week {week_number}: No working days.")
                    continue

                daily_volumes = self.daily_goals[working_indices, machine_index]

                # if self.verbose:
                #    print(f"Week {week_number} before adjustment: {daily_volumes}")

                # Check if any daily volume is less than min_demand
                if np.any(daily_volumes < min_demand) and daily_volumes.sum() > 0:
                    weekly_demand = daily_volumes.sum()

                    # Calculate number of days needed
                    num_days_needed = int(np.ceil(weekly_demand / min_demand))
                    num_days_available = len(working_indices)

                    num_days_needed = min(num_days_needed, num_days_available)

                    new_daily_volume = weekly_demand / num_days_needed

                    # if self.verbose:
                    # print(f"Week {week_number}: Reassigning weekly demand {weekly_demand} over "
                    #      f"{num_days_needed} days with new daily volume {new_daily_volume}")

                    # Assign the new daily volumes
                    self.daily_goals[working_indices, machine_index] = 0
                    self.daily_goals[working_indices[:num_days_needed], machine_index] = new_daily_volume

                # if self.verbose:
                # print(f"Week {week_number} after adjustment: {self.daily_goals[working_indices, machine_index]}")

    def get_production_schedule(self):
        if self.daily_goals is None:
            raise ValueError("You must redistribute the demand before accessing the production schedule.")

        # Create a DataFrame with the results
        self.df = pd.DataFrame({
            'Date': self.date_range,
            'Holiday': self.holiday_flags
        })

        # Add volume goals per machine using custom machine names
        for idx, machine_name in enumerate(self.machine_names):
            self.df[f'{machine_name} Goal'] = self.daily_goals[:, idx]

        if self.verbose:
            print("\nFinal Production Schedule (head 25):")
            with pd.option_context('display.max_columns', None, 'display.width', None):
                print(self.df.head(25))
        return self.df

    def get_production_plan_without_holidays(self, start_date):
        if self.daily_goal_per_machine is None or self.days_to_complete is None:
            raise ValueError(
                "You must calculate the production plan before getting the production plan without redistribution.")

        # Calculate the maximum number of days required based on the longest time needed by any machine
        num_days = max(self.days_to_complete)

        # Create a date range starting from the start_date
        date_range = pd.date_range(start=start_date, periods=num_days, freq='D')

        # Initialize a DataFrame without considering holidays or redistributions
        df_without_holidays = pd.DataFrame({
            'Date': date_range
        })

        # Add volume goals per machine
        for idx, machine_name in enumerate(self.machine_names):
            machine_goal = []
            remaining_volume = self.assigned_volume_per_machine[idx]
            daily_goal = self.daily_goal_per_machine[idx]

            for day in range(num_days):
                if remaining_volume <= 0:
                    machine_goal.append(0)
                else:
                    if remaining_volume < daily_goal:
                        # Assign all remaining volume to the last day
                        daily_production = remaining_volume
                    else:
                        daily_production = daily_goal

                    machine_goal.append(daily_production)
                    remaining_volume -= daily_production

            df_without_holidays[f'{machine_name} Goal'] = machine_goal

        # Add Holiday = False column to avoid ValueError in other operations
        df_without_holidays['Holiday'] = False

        if self.verbose:
            print("\nProduction Plan Without Holidays (head 25):")
            print(df_without_holidays.head(25))

        return df_without_holidays


# Example Usage:
"""
assigned_volume_per_machine = {1:10000,2:6000,3:10000} 
total_volume = sum(assigned_volume_per_machine)  # t
production_volume_per_day = 1200  # t/day
machine_ids = [1, 2, 3]

# Step 1: Initialize the ProductionPlanning class
planning = ProductionPlanning(total_volume, production_volume_per_day, machine_ids, assigned_volume_per_machine,
                              verbose=True)

# Step 2: Calculate production plan with verbose output
planning.calculate_production_plan()

# Step 3: Get production plan without holidays or redistribution
start_date = '2024-09-02'  # Start date for production
df_without_holidays = planning.get_production_plan_without_holidays(start_date)

print("\nRedistribute Holiday Demand")

# Step 4: Redistribute demand considering holidays
start_date = '2024-09-02'  # Monday, 2nd of September 2024
holidays = pd.to_datetime(['2024-09-17'])  # Example holidays
planning.redistribute_holiday_demand(start_date, holidays)
df = planning.get_production_schedule()

print("\nAdjust Minimum Daily Demand")

# Step 5: Adjust the daily goals to ensure minimum daily volume
planning.adjust_minimum_daily_demand(min_demand=4000)

# Step 6: Get and display the final production schedule DataFrame
df = planning.get_production_schedule()
"""
