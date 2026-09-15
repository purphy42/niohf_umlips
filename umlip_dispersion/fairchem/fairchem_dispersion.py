#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit
    from fairchem.core import FAIRChemCalculator
except ImportError:
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit

from common import forces_from_ase_calc, run_dispersion

pot_path = "/home/a.burov/umlip/fairchem/uma-s-1.pt"
predict_unit = load_predict_unit(pot_path, device="cpu")
try:
    calc = FAIRChemCalculator(predict_unit, task_name="omat")
except TypeError:
    calc = FAIRChemCalculator(predict_unit)

def force_fn(atoms):
    return forces_from_ase_calc(atoms, calc)

if __name__ == "__main__":
    run_dispersion("fairchem", force_fn, Path(__file__).resolve().parent)
