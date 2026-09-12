"""Grouped verification entry; writes evidence, never reruns annual optimization."""
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    subprocess.run([sys.executable,str(ROOT/'project.py'),'verify',*sys.argv[1:]],cwd=ROOT,check=True)
