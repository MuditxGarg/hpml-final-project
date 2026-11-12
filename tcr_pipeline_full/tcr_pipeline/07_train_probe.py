"""Corrected nested evaluation entry point; historical implementation in legacy/."""
import runpy
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.argv.append("--require-hidden")
    runpy.run_path(str(Path(__file__).with_name('09_evaluate.py')), run_name='__main__')
