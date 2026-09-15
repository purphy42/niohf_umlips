#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != _here]
sys.path.insert(0, str(_here.parent))

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.model import CHGNet
from common import run_dispersion

model = CHGNet.load(model_name="0.3.0", use_device="cpu")

def force_fn(atoms: Atoms):
    pred = model.predict_structure(AseAtomsAdaptor.get_structure(atoms))
    forces = pred.get("f", pred.get("forces"))
    if forces is None:
        raise KeyError("CHGNet prediction has neither 'f' nor 'forces'")
    return np.array(forces, dtype=float)

if __name__ == "__main__":
    run_dispersion("chgnet", force_fn, _here)
