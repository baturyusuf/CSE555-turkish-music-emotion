import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--target", type=str, default="Class")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scripts = [
        "09_sensitivity_analysis.py",
        "10_report_quality_figures.py",
        "11_clustering_sensitivity.py",
        "12_statistical_testing_enhanced.py",
        "13_mathematical_support_tables.py",
    ]

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) if not existing else str(root) + os.pathsep + existing

    for script in scripts:
        cmd = [sys.executable, str(root / "scripts" / script), "--target", args.target]
        if args.csv:
            cmd.extend(["--csv", args.csv])
        print("\n" + "=" * 96)
        print(f"RUNNING {script}")
        print("=" * 96)
        subprocess.run(cmd, check=True, cwd=str(root), env=env)


if __name__ == "__main__":
    main()
