#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_DIR = "/home/a.burov/soft/umlip/potentials"
from tace.interface.ase import TACEAseCalc
from common import forces_from_ase_calc, run_dispersion

tace_cal = TACEAseCalc(
    model=f"{MODEL_DIR}/TACE-OMAT24-L.pt",
    dtype="float32",
    device="cpu",
    fidelity_idx=0,
)

def force_fn(atoms):
    return forces_from_ase_calc(atoms, tace_cal)

if __name__ == "__main__":
    run_dispersion("tace", force_fn, Path(__file__).resolve().parent)
