#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorpotential.calculator.foundation_models import grace_fm
from common import forces_from_ase_calc, run_dispersion

calc = grace_fm(
    "GRACE-2L-OMAT-large-ft-AM",
    pad_atoms_number=10,
    pad_neighbors_fraction=0.05,
    min_dist=0.5,
)

def force_fn(atoms):
    return forces_from_ase_calc(atoms, calc)

if __name__ == "__main__":
    run_dispersion("grace", force_fn, Path(__file__).resolve().parent)
