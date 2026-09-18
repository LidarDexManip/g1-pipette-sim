"""Compatibility entry point; see tools/illustrations/arm_perturbations.py."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.illustrations.arm_perturbations import main

if __name__ == "__main__":
    main()
