#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_DIR = "/home/a.burov/soft/umlip/potentials"
from GGNN.common.calculator import UCalculator
from common import forces_from_ase_calc, run_dispersion

ggnn_cal = UCalculator(
    checkpoint_path=f"{MODEL_DIR}/equflash-OMat24.pt",
    cpu=True,
)

def force_fn(atoms):
    return forces_from_ase_calc(atoms, ggnn_cal)

if __name__ == "__main__":
    run_dispersion("ggnn", force_fn, Path(__file__).resolve().parent)
