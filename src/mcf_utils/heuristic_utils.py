import math
import pandas as pd
from typing import List, Dict, Any, Optional, Union

def warn_red(message):
    print(f"\033[91m{message}\033[0m")  # 91 is the ANSI code for red text


# Helper functions

# OLD
def create_harvest_schedule(route, beet_volumes, productivity, max_working_hours_per_period, break_duration):
    """
    Creates a harvest schedule.
    :param route: List of int, positions to harvest.
    :param beet_volumes: array of volume of beets at each position in the route.
    :param productivity: int, tons per hour.
    :param max_working_hours_per_period: int, max hours of work before a break period.
    :param break_duration: int, duration of the break in hours.
    :return: list, harvest schedule of positions per time unit.
    """

    schedule = []

    for field in route[1:]:
        volume = beet_volumes[field - 1]
        work_needed = (math.ceil(volume / productivity))
        schedule.extend([field] * work_needed)  # Stay for work_needed periods

    current_position = 0
    work_periods = 0
    result = [[0, 0]]

    for i in range(len(schedule)):
        next_position = schedule[i]

        # If it's a special case where direct movement is more efficient
        if i > 0 and schedule[i] != schedule[i - 1]:
            if work_periods > 0:
                result.append([current_position, 0])  # Move back to depot
                result.extend([[0, 0]] * break_duration)
                current_position = 0
                work_periods = 0

        # Move to the next position if not already there
        if current_position != next_position:
            result.append([current_position, next_position])
            current_position = next_position
            work_periods += 1

        # Work at the current position
        result.append([current_position, current_position])
        work_periods += 1

        # Check if a break is needed
        if work_periods == max_working_hours_per_period - 1:
            result.append([current_position, 0])  # Move back to depot
            result.extend([[0, 0]] * break_duration)
            current_position = 0
            work_periods = 0

    # Final movement to the depot
    if result[-1][1] != 0:  # Check if last position is not already the depot
        result.append([result[-1][1], 0])

    # Remove trailing [0, 0] values
    while result and result[-1] == [0, 0]:
        result.pop()

    result.extend([[0, 0]])

    return result


# OLD
def create_partial_travel_harvest_schedule(route, beet_volumes, productivity, tau):
    """
    Creates a harvest schedule without breaks.
    :param route: List of int, positions to harvest.
    :param beet_volumes: List of int, volume of beets at each position in the route.
    :param productivity: int, tons per hour.
    :param tau: np.array, travel times between fields.
    :return: list, harvest schedule of positions per time unit.
    """
    schedule = [[0, 0]]  # Start at the depot
    total_time = 0

    for i in range(len(route) - 1):
        if i == 0:
            schedule.append([route[0], route[1]])
            continue
        current_field = route[i]
        next_field = route[i + 1]

        # Calculate work needed for the current field
        volume = beet_volumes[current_field - 1]
        work_needed = volume / productivity  # Work needed in hours

        # Calculate travel time to the next field
        travel_time = tau[current_field, next_field]

        # Work done during travel
        work_done_during_travel = productivity * travel_time

        if work_done_during_travel >= volume:
            # If work is completed during travel, move directly to the next field
            schedule.append([current_field, next_field])
        else:
            # Add remaining work periods for the current field until completion
            remaining_work_needed = work_needed - travel_time

            while remaining_work_needed > 0:
                schedule.append([current_field, current_field])
                total_time += 1
                remaining_work_needed -= 1

            # After completing the work, move to the next field
            schedule.append([current_field, next_field])

    # Final movement to the depot if not already there
    if schedule[-1][1] != 0:
        schedule.append([schedule[-1][1], 0])

    return schedule


# OLD
def create_production_plan_schedule(route, beet_volumes, productivity, daily_limits, time_period_length):
    """
    Creates a harvest schedule with movements and work.
    :param route: List of int, positions to harvest.
    :param beet_volumes: List of volumes of beets at each position in the route.
    :param productivity: int, tons per hour.
    :param daily_limits: List of int, max tons that can be harvested each day.
    :param time_period_length: int, duration of each time period in hours.
    :return: list, harvest schedule of movements and work.
    """

    schedule = [[0, 0]]
    p = productivity  # Productivity got scaled in instance data

    current_position = 0  # Set starting position
    work_periods = 0
    current_day = 0
    current_daily_volume = 0

    for field in route:
        if field == 0:  # Skip depot
            continue

        volume = beet_volumes[field - 1]  # Adjust index for 0-based array
        remaining_volume = volume

        while remaining_volume > 0:
            # Get the daily limit for the current day
            if current_day < len(daily_limits):
                daily_limit = daily_limits[current_day]
            else:
                daily_limit = daily_limits[-1]  # Use the last limit if we run out of days

                # Add the check here
                if daily_limit == 0:
                    print("Warning: Daily limit is 0 on day", current_day,
                          "- breaking before finishing as no more harvesting can be done.")
                    break  # Exit the loop to prevent infinite looping

            # Calculate available capacity for today
            available_capacity = daily_limit - current_daily_volume

            # TODO: change if idle time should be triggered currently "[[current_position, current_position]]"

            # If no capacity is left for today, insert idle time and reset for the next day
            if available_capacity <= 0:
                schedule.extend(
                    [[current_position, current_position]] * (24 // time_period_length - work_periods))  # Idle
                current_day += 1
                current_daily_volume = 0
                work_periods = 0
                continue  # Move to the next day but stay on the same field

            # Work on the current field
            work_done = min(p, remaining_volume, available_capacity)  # Ensure we don't exceed daily limit
            periods_to_work = math.ceil(work_done / p)

            # Move to the field if not already there
            if current_position != field:
                schedule.append([current_position, field])
                current_position = field
                work_periods += 1

            # Perform the work
            schedule.extend([[current_position, current_position]] * periods_to_work)
            current_daily_volume += work_done
            remaining_volume -= work_done
            work_periods += periods_to_work

            # Check if the field is done
            if remaining_volume <= 0:
                if field != route[-1]:  # If there's another field, move to it
                    next_field = route[route.index(field) + 1]
                    schedule.append([current_position, next_field])
                    current_position = next_field
                    work_periods += 1
                break  # Move to the next field in the outer loop

    # Final movement to the depot if not already there
    if schedule[-1][1] != 0:
        schedule.append([schedule[-1][1], 0])

    return schedule


# Used currently
def create_production_plan_schedule_tau(
    route: List[int],
    beet_volumes: Union[Dict[int, float], List[float], pd.Series],
    productivity: float,
    daily_limits: List[float],
    time_period_length: int,
    tau: Union[pd.DataFrame, Dict[int, pd.DataFrame]],
    *,
    machine_id: Optional[int] = None,
    vehicle_capacity_flag: bool = False,
    time_restriction: bool = True,
    verbose: bool = False
) -> List[List[Union[int, str]]]:
    """
    Create a period‐by‐period loading schedule for a single loader following a given route.

    This heuristic:
      1. Starts and ends each day at the depot (node 0), then visits fields in `route` order.
      2. In each time period:
         - Applies up to `productivity` tons of loading to the current field,
           including partial work during travel time (1 – travel_time fraction).
         - Observes a per‐day loading quota (`daily_limits`), splitting field loading
           across days when necessary.
         - Inserts idle slots on full‐quota days or holidays (zero‐limit days).
      3. Advances to the next field when its volume is exhausted, or the day ends.
      4. Stops once all fields are loaded or all days in `daily_limits` are used
         (if `time_restriction=True`), then returns to the depot.

    Args:
        route: Ordered list of field IDs to visit (must start/end with 0).
        beet_volumes: Mapping or sequence of total beet volumes per field ID.
        productivity: Tons loadable per hour in a work slot.
        daily_limits: Maximum tons loadable each day (one entry per day).
        time_period_length: Hours per time slot (e.g., 2 for 2-hour slots).
        tau: Travel time matrix (DataFrame) or dict of matrices keyed by `machine_id`.
        machine_id: If `tau` is a dict, the key for selecting this loader’s matrix.
        vehicle_capacity_flag: If True, forbids loading once daily limit is hit.
        time_restriction: If True, halts scheduling when `daily_limits` are exhausted.
        verbose: If True, prints detailed logs of days, fields, and idle periods.

    Returns:
        A list of `[from_node, to_node]` pairs for each time period—covering
        work (same→same), travel, idle (same→same on quota/holidays)—
        ending with a final return‐to‐depot move.
    """

    schedule = [[0, 0]]  # Start at the depot

    # Initialise tracking variables
    work_periods = 0
    current_day = 0
    current_daily_volume = 0
    total_days = len(daily_limits)
    periods_per_day = 24 // time_period_length  # e.g., 12 periods for 2-hour periods

    # Iterate over route to respect sequence
    for i in range(len(route) - 1):
        if verbose:
            print(f"\nProcessing field {route[i]} (index {i})")

        if work_periods >= periods_per_day:
            print("Work limit triggered")
            # Move to next day and reset tracking
            current_day += 1
            work_periods = 0
            current_daily_volume = 0

        # Pre-Day Check: Handle Holidays
        while current_day < total_days and daily_limits[current_day] == 0:
            if verbose:
                print(f"Day {current_day + 1} is a holiday. Skipping to the next day.")
            # Insert idle periods for the entire day
            schedule.extend([[f"{current_position}", f"{current_position}"]] * periods_per_day)
            current_day += 1
            current_daily_volume = 0
            work_periods = 0

        # Check if all days are processed
        if time_restriction:
            if current_day >= total_days:
                warn_red("Warning: All days have been processed. Cannot schedule more work.")
                break

        # When starting at depot
        if i == 0:
            # Before moving, check if moving is allowed today
            if daily_limits[current_day] == 0:
                # Insert idle periods until next available day
                while current_day < total_days and daily_limits[current_day] == 0:
                    if verbose:
                        print(f"Day {current_day + 1} is a holiday. Skipping to the next day.")
                    schedule.extend([[f"{current_position}", f"{current_position}"]] * periods_per_day)
                    current_day += 1
                    current_daily_volume = 0
                    work_periods = 0
                if time_restriction:
                    if current_day >= total_days:
                        warn_red("Warning: All days have been processed. Cannot schedule more work.")
                        return schedule
            # Now we can move
            schedule.append([route[0], route[1]])
            work_periods += 1
            continue

        current_field = route[i]
        current_position = current_field
        next_field = route[i + 1]

        # Calculate work needed for the current field
        volume = beet_volumes[current_field]

        # Calculate travel time to the next field in hours
        if machine_id:
            travel_time = tau[machine_id].at[current_field, next_field]
        else:
            travel_time = tau.at[current_field, next_field]

        # Work done during travel (rest of time and only movement to field not from field)
        max_work_done_during_travel = productivity * max(0, (1 - travel_time))

        # Get the daily limit for the current day
        if current_day < len(daily_limits)-1:
            daily_limit = daily_limits[current_day]

        else:
            try:
                daily_limit = daily_limits[-2]  # Try to use the second-to-last limit
            except IndexError:
                daily_limit = daily_limits[-1]  # Fallback to the last limit if IndexError occurs

            # Prevent infinite loop
            if daily_limit == 0:
                warn_red(
                    f"Warning: Daily limit is 0 on day {current_day} - "
                    f"breaking before finishing as no more harvesting can be done.")
                break

        # Calculate available capacity for today
        available_capacity = daily_limit - current_daily_volume
        # Get work done during travel
        work_done_during_travel = min(volume, max_work_done_during_travel, available_capacity)
        # Subtract work during travel from volume at field
        remaining_volume = volume - work_done_during_travel
        # Update daily volume counter
        current_daily_volume += work_done_during_travel
        # Update work_period counter

        if work_done_during_travel >= volume:
            # If work is completed during travel, move directly to the next field
            schedule.append([current_field, next_field])
            # Update work_period counter
            work_periods += 1

            if work_periods >= periods_per_day:
                print("Work limit triggered")
                # Move to next day and reset tracking
                current_day += 1
                work_periods = 0
                current_daily_volume = 0

        while remaining_volume > 0:

            # Get the daily limit for the current day
            if current_day < len(daily_limits)-1:
                daily_limit = daily_limits[current_day]
            else:
                try:
                    daily_limit = daily_limits[-2]  # Try to use the second-to-last limit
                except IndexError:
                    daily_limit = daily_limits[-1]  # Fallback to the last limit if IndexError occurs

            # Skip holidays
            while current_day < total_days and daily_limit == 0:
                if verbose:
                    print(f"Day {current_day + 1} is a holiday. Skipping to the next day.")
                # Insert idle periods for the entire day
                schedule.extend([[f"{current_field}", f"{current_position}"]] * (periods_per_day - work_periods))
                current_day += 1
                work_periods = 0
                current_daily_volume = 0
                if current_day >= total_days:
                    warn_red("Warning: All days have been processed. Cannot complete finishing all fields.")
                    if schedule[-1][1] != 0:
                        schedule.append([schedule[-1][1], 0])
                    return schedule  # Return the schedule up to this point

                if current_day < len(daily_limits) - 1:
                    daily_limit = daily_limits[current_day]

                else:
                    try:
                        daily_limit = daily_limits[-2]  # Try to use the second-to-last limit
                    except IndexError:
                        daily_limit = daily_limits[-1]  # Fallback to the last limit if IndexError occurs


            # Calculate available capacity for today
            available_capacity = daily_limit - current_daily_volume
            # If no capacity is left for today, insert idle time and reset for the next day
            if available_capacity <= 0 or work_periods >= periods_per_day:
                schedule.extend(
                    [[f"{current_field}", f"{current_field}"]] * ((24 // time_period_length) - work_periods))  # Idle
                current_day += 1
                current_daily_volume = 0
                work_periods = 0

                if time_restriction:
                    if vehicle_capacity_flag:
                        if current_day >= total_days:
                            warn_red("Warning (Expected due to over-assignment): "
                                     "No more capacity available. Cannot complete finishing all fields.")
                            if schedule[-1][1] != 0:
                                schedule.append([schedule[-1][1], 0])
                            return schedule  # Return the schedule up to this point

                continue  # Move to the next day but stay on the same field

            # Work on the current field
            work_done = min(productivity, remaining_volume, available_capacity)  # Ensure we don't exceed daily limit
            periods_to_work = math.ceil(work_done / productivity)

            # Move to the field if not already there
            if current_position != route[i]:
                schedule.append([current_position, route[i]])
                current_position = route[i]
                work_periods += 1

            # Perform the work
            schedule.extend([[current_position, current_position]] * periods_to_work)
            current_daily_volume += work_done
            remaining_volume -= work_done
            work_periods += periods_to_work

            # Check if the field is done
            if remaining_volume <= 0:

                if route[i] != route[-1]:  # If there's another field, move to it
                    next_field = route[route.index(route[i]) + 1]

                    # Before moving, check if moving is allowed today
                    if daily_limit == 0 or work_periods >= periods_per_day:

                        # Insert idle periods until next available day
                        idle_periods = periods_per_day - work_periods
                        schedule.extend([[f"{current_position}", f"{current_position}"]] * idle_periods)
                        current_day += 1
                        work_periods = 0
                        current_daily_volume = 0

                        while current_day < total_days and daily_limits[current_day] == 0:
                            if verbose:
                                print(f"Day {current_day + 1} is a holiday. Skipping to the next day.")
                            schedule.extend([[f"{current_position}", f"{current_position}"]] * periods_per_day)
                            current_day += 1

                        if time_restriction:
                            if current_day >= total_days:
                                warn_red("Warning: All days have been processed. Cannot schedule more work.")
                                if schedule[-1][1] != 0:
                                    schedule.append([schedule[-1][1], 0])
                                return schedule
                    # Move to the next field
                    schedule.append([current_position, next_field])
                    work_periods += 1
                    current_position = next_field
                break  # Move to the next field in the outer loop

            # Update current_position after moving
        current_position = route[i + 1]

    # Final movement to the depot if not already there
    if schedule[-1][1] != 0:
        schedule.append([schedule[-1][1], 0])

    return schedule



