"""Compatibility entry point; see tools/illustrations/lab_environments.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.illustrations.lab_environments import main

if __name__ == "__main__":
    main()
