#!/usr/bin/env python3
# coding: utf-8
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _here]
sys.path.insert(0, str(_here.parent))

import warnings
import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from m3gnet.models import Relaxer
from common import run_dispersion

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

relaxer = Relaxer(potential="MP-2021.2.8-EFS")

def force_fn(atoms: Atoms):
    efs = relaxer.potential.get_efs(AseAtomsAdaptor.get_structure(atoms))
    if isinstance(efs, dict):
        forces = efs.get("f", efs.get("forces"))
    else:
        forces = efs[1]
    forces = np.asarray(forces, dtype=float)
    if forces.ndim == 3:
        forces = forces[0]
    return forces

if __name__ == "__main__":
    run_dispersion("m3gnet", force_fn, _here)
