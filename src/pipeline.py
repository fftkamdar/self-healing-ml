
from pathlib import Path
import subprocess
import sys


# ============================================================
# SELF-HEALING ML
# Automated Monitoring and Retraining Pipeline
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

DRIFT_SCRIPT = SRC_DIR / "drift_monitor.py"
RETRAIN_SCRIPT = SRC_DIR / "retrain_experiment.py"


def run_step(step_name, script_path):
    """Run one pipeline step and stop if it fails."""

    print("\n" + "=" * 65)
    print(f"STARTING: {step_name}")
    print("=" * 65)

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed.")
        print("Return code:", result.returncode)
        return False

    print(f"\nCOMPLETED: {step_name}")
    return True


def main():
    print("=" * 65)
    print("SELF-HEALING ML")
    print("Automated Drift Monitoring and Retraining Pipeline")
    print("=" * 65)

    print("\nProject directory:", PROJECT_ROOT)

    # Step 1: Monitor feature-level drift.
    if not run_step(
        "Feature-Level Drift Monitoring",
        DRIFT_SCRIPT,
    ):
        print("\nPipeline stopped because drift monitoring failed.")
        sys.exit(1)

    # Step 2: Apply the configured retraining decision rule.
    if not run_step(
        "Drift-Triggered Retraining Experiment",
        RETRAIN_SCRIPT,
    ):
        print("\nPipeline stopped because the retraining experiment failed.")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nCheck these output folders:")
    print("- reports/")
    print("- models/")

    print(
        "\nNote: This pipeline runs a controlled simulated-drift "
        "experiment. It is not yet connected to a live data stream."
    )


if __name__ == "__main__":
    main()