#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from sevenn.sevennet_calculator import SevenNetCalculator
except ImportError:
    from sevenn.calculator import SevenNetCalculator
from common import forces_from_ase_calc, run_dispersion

calc = SevenNetCalculator(model="7net-mf-ompa", device="cpu", modal="mpa")

def force_fn(atoms):
    return forces_from_ase_calc(atoms, calc)

if __name__ == "__main__":
    run_dispersion("sevennet", force_fn, Path(__file__).resolve().parent)
