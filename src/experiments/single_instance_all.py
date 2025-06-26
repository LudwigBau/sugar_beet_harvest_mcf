# src/experiments/single_instance_all.py
"""
Run the four single-instance experiment scripts in sequence, using the same
Python interpreter that launched this file.
"""

import subprocess
import sys
from pathlib import Path

def run(module: str) -> None:
    """Run `python -m <module>` with the current interpreter and stop on error."""
    print(f"→ running {module}")
    subprocess.run([sys.executable, "-m", module], check=True)
    print(f"✓ finished {module}\n")

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    print(f"Executing scripts from: {here}")

    # modules, not file paths
    modules = [
        "src.experiments.single_instance_triple_sim",
        "src.experiments.single_instance_double_sim",
        "src.experiments.single_instance_single_sim",
        "src.experiments.robustness_checks",
    ]

    for mod in modules:
        run(mod)

    print("All experiment scripts completed successfully.")
