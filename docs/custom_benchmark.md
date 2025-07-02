# Customizing the Benchmark Heuristic

This guide explains how to modify the **benchmark heuristic** used in the sugar beet MCF model experiments.

The benchmark is a drop-in replacement for the default heuristic and is invoked in every experiment to generate loading and production schedules. It provides flexibility for users to prototype and evaluate new strategies while maintaining full compatibility with the experimental pipeline.

---

## 1. Locate the Heuristic Function

The default benchmark heuristic is implemented in the following file:

```
src/mcf_utils/heuristic_utils.py
```


Open this file in your editor and scroll to the function:

> *All type hints follow the `typing` module introduced in Python 3.5
> (PEP 484)--e.g. `List[int]`, `Dict[str, float]`.*

```python
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
```

This function defines a time-indexed schedule for one machine. Each entry in the output list represents a period, containing either a location (int) or the string "idle".


### Why your heuristic must mimic `create_production_plan_schedule_tau`

`enforce_schedule_with_constraints_pruned` converts the schedule that your heuristic emits into **hard Gurobi constraints**.
That routine expects, for every loader `h` and every slot `t`, exactly one arc `(i,j)` to be fixed to one, all others to zero.
If your heuristic returns anything that does not match the shape or the semantics produced by `create_production_plan_schedule_tau`, the constraint builder will either throw an index error or, worse, will relax infeasible moves and the model will silently mis-count capacity. The safest way to avoid that risk is to keep **both the interface *and* the behavioural rules identical** to the reference function. ([github.com][1])

**Interface contract**

| Argument             | Meaning (short)                        | What your heuristic must honour                                          |
| -------------------- | -------------------------------------- | ------------------------------------------------------------------------ |
| `route`              | ordered list of node IDs, depot is `0` | output must follow this sequence and add the final depot return          |
| `beet_volumes`       | tons per field                         | field is finished if cumulative work equals this volume before moving on |
| `productivity`       | tons per hour loaded                   | used to bound work per slot                                              |
| `daily_limits`       | max tons loaded each day               | heuristic must stop work or insert idle once the day-quota is hit        |
| `time_period_length` | hours per slot                         | determines how many slots a day contains (`24 / length`)                 |
| `tau`                | travel time in hours                   | work during travel is limited to `productivity × (1 – travel_time)`      |

The **return value** is a Python list whose `k`-th entry is `[from_node_k, to_node_k]`.

* `schedule[0]` is always `[0, 0]` (begin at depot).
* The last element must end at depot.
* Idle or pure work slots are coded as self-loops `[i, i]`.
* Length of the list equals the number of time periods actually used.

---

### A minimal worked example

Suppose you want to harvest two fields with the following data.

| Parameter                  | Value                                         |
| -------------------------- | --------------------------------------------- |
| Route                      | `[0, 1, 2, 0]`                                |
| Beet volumes               | `{1: 30, 2: 50}` t                            |
| Loader productivity        | `10` t per h                                  |
| Length of one slot         | `2` h ⇒ `20` t max per slot                   |
| Daily limits               | `[40, 40]` t for the first two days           |
| Travel time matrix `τ (h)` | `0→1: 0.5`, `1→2: 0.5`, `2→0: 0.5`, symmetric |

`periods_per_day = 24 / 2 = 12`.
`max_work_in_travel = productivity × (1 – τ) = 10 × (1 – 0.5) = 5` t while moving.

**Day 1**

| Slot | Move         | Tons loaded today (cumulative) | Comment                                                  |
| ---- |--------------| ------------------------------ |----------------------------------------------------------|
| 0    | `[0, 1]`     | 5                              | travel to field 1 and load 5 t during the remaining hour |
| 1    | `[1, 1]`     | 25                             | work 20 t at field 1                                     |
| 2    | `[1, 1]`     | 30                             | finish the remaining 5 t (field 1 is now empty)          |
| 3    | `[1, 2]`     | 35                             | travel to field 2 and load 5 t on the way                |
| 4    | `[2, 2]`     | 40                             | load 5 t at field 2, daily quota reached                 |
| 5-11 | `['2', '2']` | 40                             | idle self-loops until midnight indicated with str(i)     |

**Day 2**

Reset the daily counter, continue at field 2.

| Slot | Move     | Daily load | Comment                                           |
|------| -------- | ---------- |---------------------------------------------------|
| 0-2  | `[2, 2]` | 60         | three full work slots load 60 t and clear field 2 |
| 3    | `[2, 0]` | 60         | travel back to depot, schedule finished           |
| 4    | `[0, 0]` | 60         | stay at depot                 |

**Resulting schedule list**

```python
schedule = [
    [0, 0],         # initial position
    [0, 1],         # travel with partial work
    [1, 1],         # work
    [1, 1],         # work
    [1, 2],         # travel to next field
    [2, 2],         # work
    ['2', '2'],     # idle rest of day (self-loops)
    ...
    [2, 2],         # start of day 2, work field 2
    [2, 2],
    [2, 2],
    [2, 0],         # return to depot
    [0, 0]          # end with depot stay
]
```

Every field is cleared before the loader advances, and the 40 t daily ceiling is never breached.

---

### How to plug in your own heuristic

The reference heuristic mirrors the way human dispatchers assign a loader to one field at a time, so every schedule it generates is both **feasible** and **close to real-life practice**. If you would like to raise the performance bar you are welcome to replace its logic, **but keep the same function name, inputs, and outputs**—that is what the rest of the pipeline expects.

* **Inputs (typing recap)**

  * `route : List[int]` – ordered node IDs, depot `0` first
  * `beet_volumes : Dict[int, float] | List[float] | pd.Series` – tons per field
  * `productivity : float` – tons per hour a loader can handle
  * `daily_limits : List[float]` – *optional* daily quotas; may be empty or ignored
  * `time_period_length : int` – hours per discrete time slot (1h)
  * `tau : pd.DataFrame | Dict[int, pd.DataFrame]` – travel-time matrix (or one per machine)
  * keyword flags as in the original signature


* **Output (schedule list)**

  * A list whose *k*-th element is `[from_node, to_node]`.
  * Begin with `[0, 0]`; end with a move that returns to depot `0`.
  * **Work periods** are self-loops `[i, i]` (both integers).
  * **Idle periods** are self-loops with *strings*: `[str(i), str(i)]`.
    The enforcement layer uses this distinction to fix the correct binary variables.
  * Length equals the number of time slots you actually schedule.


* **Industrial rules that must hold**

  1. A field can be left only after its assigned volume is fully loaded.
  2. Travel duration between nodes follows `tau`.
  3. Every loader finishes the horizon at the depot.
  4. Daily limits may be ignored, but any other hard constraints in the model must be respected.

If your custom logic honours these points, `enforce_schedule_with_constraints_pruned` will treat the output exactly like the benchmark schedule, giving you a clean apples-to-apples comparison.

