import subprocess
import sys
from pathlib import Path


def run_script(mod_name: str):
    print(f"\n--- Running {mod_name} ---")
    subprocess.run([sys.executable, "-m", mod_name], check=True)
    print(f"--- Finished {mod_name} ---\n")


def main():
    repo_root = Path(__file__).resolve().parent.parent

    experiments = [
        "src.experiments.single_instance_all",
        "src.experiments.main_loading_rolling_sim",
        "src.experiments.robustness_checks",
    ]

    for mod in experiments:
        run_script(mod)


if __name__ == "__main__":
    main()
