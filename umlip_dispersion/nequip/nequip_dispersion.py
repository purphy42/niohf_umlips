#!/usr/bin/env python3
# coding: utf-8
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nequip.ase import NequIPCalculator
from common import forces_from_ase_calc, run_dispersion

_nequip_pt = "/home/a.burov/umlip/nequip/mir-group__NequIP-OAM-L__0.1.nequip.pt2"
_species = {"Ni": "Ni", "O": "O", "H": "H", "F": "F"}
try:
    calc = NequIPCalculator.from_compiled_model(
        compile_path=_nequip_pt,
        device="cpu",
        chemical_species_to_atom_type_map=_species,
    )
except Exception:
    calc = NequIPCalculator.from_deployed_model(
        model_path=_nequip_pt,
        device="cpu",
        species_to_type_name=_species,
    )

def force_fn(atoms):
    return forces_from_ase_calc(atoms, calc)

if __name__ == "__main__":
    run_dispersion("nequip", force_fn, Path(__file__).resolve().parent)
