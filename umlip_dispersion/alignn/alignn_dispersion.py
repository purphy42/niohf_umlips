#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
from common import forces_from_ase_calc, run_dispersion

model_path = "/home/a.burov/umlip/alignn/v12.2.2024_mp_1.5mill/"
if not Path(model_path).exists():
    model_path = default_path()
    print(f"ALIGNN cluster path missing; using default_path={model_path}")
calc = AlignnAtomwiseCalculator(path=model_path, model_filename="best_model.pt")

def force_fn(atoms):
    return forces_from_ase_calc(atoms, calc)

if __name__ == "__main__":
    run_dispersion("alignn", force_fn, Path(__file__).resolve().parent)
