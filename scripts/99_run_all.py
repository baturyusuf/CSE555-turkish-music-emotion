import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    '01_data_quality.py',
    '02_correlation_analysis.py',
    '03_feature_discriminability.py',
    '04_pca_analysis.py',
    '05_lda_analysis.py',
    '06_clustering_analysis.py',
    '07_hypothesis_testing.py',
    '08_manual_calculation_support.py',
]


def main():
    base = Path(__file__).resolve().parent
    extra_args = sys.argv[1:]
    for script in SCRIPTS:
        cmd = [sys.executable, str(base / script)] + extra_args
        print('\n' + '=' * 90)
        print('RUNNING', script)
        print('=' * 90)
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
