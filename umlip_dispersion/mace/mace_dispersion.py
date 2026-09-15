#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mace.calculators import mace_mp
from common import forces_from_ase_calc, run_dispersion

calc = mace_mp(model="large", device="cpu", default_dtype="float64", dispersion=False)

def force_fn(atoms):
    return forces_from_ase_calc(atoms, calc)

if __name__ == "__main__":
    run_dispersion("mace", force_fn, Path(__file__).resolve().parent)
