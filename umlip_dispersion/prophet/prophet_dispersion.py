#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_DIR = "/home/a.burov/soft/umlip/potentials"
from prophet import KairosCalculator
from common import forces_from_ase_calc, run_dispersion

prophet_cal = KairosCalculator(
    model_path=f"{MODEL_DIR}/prophet-oame-mbd.pt",
    use_kernel=False,
    device="cpu",
)

def force_fn(atoms):
    return forces_from_ase_calc(atoms, prophet_cal)

if __name__ == "__main__":
    run_dispersion("prophet", force_fn, Path(__file__).resolve().parent)
