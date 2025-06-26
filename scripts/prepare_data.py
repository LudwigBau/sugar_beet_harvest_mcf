import sys
import subprocess
from pathlib import Path

# ===== CONFIGURATION =====
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SIM_DATA_DIR = DATA_DIR / "simulated_data"
FIGURES_DIR = DATA_DIR / "figures"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_SUBDIRS = ["excel", "instances", "reporting"]

REQUIRED_FILES = [
    "simulated_fields.csv",
    "simulated_machine_df.csv",
    "simulated_cost_matrix_df.csv",
]

# ===== HELPERS =====
def ensure_directories():
    for d in [FIGURES_DIR, SIM_DATA_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for sub in RESULTS_SUBDIRS:
        (RESULTS_DIR / sub).mkdir(exist_ok=True)
    print("✔️  Data folder structure is in place.")


def check_manual_data_files():
    print("📁 Checking manually downloaded data files...")
    missing = [f for f in REQUIRED_FILES if not (SIM_DATA_DIR / f).exists()]
    if missing:
        print("❌ Missing the following required files in data/simulated_data/:")
        for f in missing:
            print(f"   - {f}")
        print("\n📌 Please download them from Zenodo and place them in:")
        print(f"   {SIM_DATA_DIR.resolve()}\n")
        sys.exit(1)
    print("✅ All required data files are present.")


def run_main_instance_data():
    script = ROOT / "src" / "data_prep" / "main_instance_data.py"
    if not script.exists():
        print(f"❌ ERROR: could not find {script}", file=sys.stderr)
        sys.exit(1)

    print(f"\n--- Running {script.name} ---")
    subprocess.run([sys.executable, "-m", "src.data_prep.main_instance_data"], check=True)
    print(f"--- Finished {script.name} ---\n")


# ===== ENTRY POINT =====
def main():
    ensure_directories()
    check_manual_data_files()
    run_main_instance_data()


if __name__ == "__main__":
    main()