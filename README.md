# Integrating Harvest Planning Along the Supply Chain: A Case Study From the European Sugar Beet Industry

This repository implements a multi-commodity flow (MCF) model and companion heuristic to plan sugar-beet loading, 
inventory, and production in an integrated, time-expanded network.  We benchmark against a customizable loading-schedule 
heuristic that mimics current industry practice and demonstrate double-digit logistics cost savings (15–20%) on both 
simulated and real European data.

**Key features**  
- **MCF model** (Gurobi‐solvable) for loaders & beet flows over hourly periods  
- **Rolling‐horizon framework**: weekly subproblems with multi-step coarse-to-fine  
- **Heuristic baseline**: `create_production_plan_schedule_tau()` in  
  `src/mcf_utils/heuristic_utils.py`  
- **Custom Benchmark**: drop in your own loading logic; infeasible runs produce IIS files  
- **Reproducible experiments**: data download, routing, scheduling, and MIP in one script  

## Quickstart

0. **Clone this repository**
    
    ```
   git clone https://github.com/LudwigBau/sugar_beet_mcf_public.git 
   cd sugar_beet_mcf_public
    ```

1. **Install**  
   ```
   pip install -r requirements.txt       # standard install
   pip install -e .                       # editable/developer mode
   ```
2. **Configure your Gurobi license**

    Make sure you have a valid Gurobi license installed and the `GRB_LICENSE_FILE` environment variable set.
    See the Gurobi Support page for licensing instructions: https://support.gurobi.com

3. **Download Data**

    The dataset is hosted on Zenodo (https://zenodo.org/records/15743878) under restricted access and is available exclusively for academic research.
    
    To obtain the required .csv files request access via Zenodo.
    
- After approval, manually download the following files from Zenodo:

    - simulated_fields.csv 
    - simulated_machine_df.csv 
    - simulated_cost_matrix_df.csv
    
  Save all files to the following folder in your cloned repo:
    ```
    data/simulated_data/
    ```

Once the files are in place, run:
    ```
    python scripts/prepare_data.pys/prepare_data.py
    ```


4. **(Optional) Adapt your heuristic**

    Edit the loader schedule heuristic here (more information in Section Custom Benchmark):
   ```
    src/mcf_utils/heuristic_utils.py
    ```

5. **Run full experiments**
    ```
    python scripts/run_experiments.py
    ```


## Results & Excel Export

After running `scripts/run_experiments.py`, all single‐instance and rolling‐horizon results are written to Excel 
workbooks under:

results/excel/



### Single‐instance schedules

For each `(scenario, model_version, sensitivity)` key, you will find:


results/excel/{combined\_key}\_flow\_results.xlsx



**Sheets** (standardized for every key):
- **Beet Movement**: time‐series of volumes harvested, loaded, stored, processed  
- **Machine Schedule**: period‐by‐period start/end positions (or “idle”) per loader  
- **Field Yield**: raw remaining volume per field × period  
- **Machinery Cost**: travel, operation & partial‐work costs per machine × period   
- **Revenue and Unmet Demand**: sugar‐yield, revenue, unmet‐demand per period  
- **Accounting**: weekly accounting breakdown (work, travel, inventory, penalties)  
- **Hidden Idle** *(if `tau` & `L_bar` available)*: loader‐level hidden‐idle metrics

Use these sheets directly in Excel

### Rolling‐horizon consolidation

A single workbook aggregates all weeks and compares Gurobi vs. heuristic:

results/excel/consolidated\_weekly\_results\_{rolling|groupID}.xlsx

**Worksheets**:  
- One sheet per metric (e.g. “Beet Movement”, “Machine Schedule”, “Accounting”, etc.)  
- Within each sheet, Gurobi and heuristic tables appear one after the other, separated by blank rows  
- A “KPI Comparison” sheet reports cost‐difference %, margins, unmet‐demand, time, and MIP gap  

All Excel output uses multi‐index columns where appropriate—just point your manuscript or analysis script at these 
files to reproduce every table and plot in the paper.




## Authors
Ludwig Baunach, Stefan Spinler and John Birge

## Cite
Baunach, L., Spinler, S., & Birge, J. (2025). Integrating Harvest Planning Along
the Supply Chain: A Case Study From the European Sugar Beet Industry.

## Hardware

Gurobi 11.0.0 via Python API

Intel Xeon Silver 4215 @ 2.50 GHz, 16 cores

MIPGap = 0.1%, TimeLimit = 3 600 s per run (max 7 200 s for coarse→fine)

## Custom Benchmark

All heuristic runs will now include your custom benchmark and, in case of infeasibility, an IIS file will be generated 
for diagnosis.

### 1. Locate the heuristic

```
# Open this file in your editor:
src/mcf_utils/heuristic_utils.py
```

### 2. Adapt the function

Edit the signature and body of:

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
    ...
```

– Preserve the signature and return type.
– Implement your new scheduling logic or performance counters here.

### 3. Metrics collected

When you re-run the experiments, all heuristic/benchmark results will use the 
adapted function.

Note new benchmark columns will not appear alongside the 
standard heuristic results. Please save them separately.

### 4. Infeasibility output

If a scenario becomes infeasible under your heuristic, an IIS file named

```
mcf_utils/m_iis.ilp
```

will be written for debugging.

### 5. Run the benchmarks

```
python scripts/run_experiments.py
```

Your adapted `create_production_plan_schedule_tau` will be invoked for every
(scenario, model\_version, sensitivity) tuple, and the new benchmark metrics
will be appended to `results/excel` as excel files and `results/reporting` as
pickle files.

## License

This software is made available under a custom academic-use license.  
Commercial use is strictly prohibited without prior written consent.  
See the [LICENSE](LICENSE) file for full terms.
