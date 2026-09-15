#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mattersim.forcefield import MatterSimCalculator
from common import forces_from_ase_calc, run_dispersion

calc = MatterSimCalculator(
    load_path="/home/a.burov/umlip/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
    device="cpu",
)

def force_fn(atoms):
    return forces_from_ase_calc(atoms, calc)

if __name__ == "__main__":
    run_dispersion("mattersim", force_fn, Path(__file__).resolve().parent)
