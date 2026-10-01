#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Internal phonopy methods

# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Kill warnings
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')


# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.optimize import FIRE
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import yaml
import subprocess

import shutil
import glob
from pathlib import Path

from ase.build import bulk
from ase.constraints import ExpCellFilter

import pandas as pd


# In[2]:


from ase.optimize import FIRE  
from ase import Atoms
from ase.optimize import BFGS
from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read, write

import warnings

from m3gnet.models import Relaxer, M3GNet, Potential
from pymatgen.core import Lattice, Structure

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")
    
from pymatgen.core import Lattice, Structure
from pymatgen.core import Structure
from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np



# In[ ]:





# In[3]:


from phonopy import Phonopy
from phonopy.interface.calculator import get_displacements_and_forces
from phonopy.structure.atoms import PhonopyAtoms


# In[4]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[5]:


np.set_printoptions(precision=4)


# In[6]:


# %matplotlib inline
plt.rcParams['figure.dpi'] = 450


# In[ ]:





# In[7]:


# CRITICAL: relax_cell is a Relaxer __init__ argument (default True).
# Passing relax_cell to .relax() is a no-op on classic m3gnet → every
# "volume" point collapses back to Veq and the EOS span is ~0.
m3gnet_relaxer = Relaxer(potential="MP-2021.2.8-EFS", relax_cell=False)


# In[ ]:





# In[8]:


# m3gnet_model = matgl.load_model("M3GNet-MP-2021.2.8-PES")  # NOT Relaxer!


# In[9]:


fontsize = 20


# In[10]:


env_used = "m3gnet-env"


# In[11]:


potential = "m3gnet"


# In[ ]:





# In[ ]:


def _ensure_fixed_volume_relaxer(relaxer):
    """Force atoms-only relaxation. relax_cell lives on Relaxer.__init__, not .relax()."""
    import inspect
    if getattr(relaxer, "relax_cell", None) is False:
        return relaxer
    pot = getattr(relaxer, "potential", None) or "MP-2021.2.8-EFS"
    kwargs = {"potential": pot, "relax_cell": False}
    try:
        params = inspect.signature(Relaxer.__init__).parameters
        for key, val in (("optimizer", getattr(relaxer, "optimizer", "FIRE")),
                         ("stress_weight", getattr(relaxer, "stress_weight", None))):
            if key in params and val is not None:
                kwargs[key] = val
    except (TypeError, ValueError):
        pass
    print("Rebuilding Relaxer with relax_cell=False (was True / unset)")
    return Relaxer(**kwargs)


def _m3gnet_relax_kwargs(relaxer, fmax, steps):
    """Kwargs for Relaxer.relax; also pass relax_cell=False if supported."""
    import inspect
    kwargs = {"fmax": fmax, "steps": steps, "verbose": False}
    try:
        params = inspect.signature(relaxer.relax).parameters
        if "relax_cell" in params:
            kwargs["relax_cell"] = False
    except (TypeError, ValueError):
        pass
    return kwargs


def _lock_volume(atoms, target_cell):
    """Restore target cell while keeping fractional coordinates."""
    frac = atoms.get_scaled_positions()
    atoms.set_cell(target_cell, scale_atoms=False)
    atoms.set_scaled_positions(frac)
    return atoms


def _phonon_fig_dir():
    for p in (
        Path("/home/arseniy/Desktop/work/niohf/figures/phonons"),
        Path("/home/a.burov/icys_2025/niohf/figures/phonons"),
        Path("/home/a.burov/icys_2025/niohf/data/figures/phonons"),
    ):
        if p.exists() or p.parent.exists():
            p.mkdir(parents=True, exist_ok=True)
            return str(p)
    p = Path("/home/arseniy/Desktop/work/niohf/figures/phonons")
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def get_forces_from_potential(relaxer, structure):
    """Forces at the given pymatgen Structure (no ionic steps)."""
    efs = relaxer.potential.get_efs(structure)
    energy, forces = efs[0], efs[1]
    energy = float(np.array(energy).reshape(-1)[0])
    forces = np.asarray(forces, dtype=float)
    if forces.ndim == 3:
        forces = forces[0]
    return forces, energy


def get_forces_from_relaxer(relaxer, structure):
    """Back-compat alias: single-shot potential forces, not a 1-step relax."""
    return get_forces_from_potential(relaxer, structure)


# In[ ]:


# In[ ]:





# In[ ]:


def calc_phonopy(equilibrium_atoms, m3gnet_relaxer, mul_matrix=[[1,0,0],[0,1,0],[0,0,1]], env_used="m3gnet-env", scale=[0.97, 1.10], volume_points=11):

    m3gnet_relaxer = _ensure_fixed_volume_relaxer(m3gnet_relaxer)
    volume_scales = np.linspace(scale[0], scale[1], volume_points)
    volumes, energies = [], []

    for f in glob.glob("thermal_properties.yaml-*") + glob.glob("gibbs_V*.dat") + glob.glob("ev_point_*.dat"):
        os.remove(f)

    print("🚀 M3GNet QHA — fixed-volume (relax_cell=False) + potential.get_efs forces")
    relax_kwargs = _m3gnet_relax_kwargs(m3gnet_relaxer, fmax=1e-4, steps=400)
    print(f"Relaxer.relax_cell={getattr(m3gnet_relaxer, 'relax_cell', '?')}  kwargs={relax_kwargs}")
    print(f"Using supercell {mul_matrix}; volume scales {scale[0]:.3f}→{scale[1]:.3f} ({volume_points} pts)")

    for i, scale_factor in enumerate(volume_scales):
        print(f"\nVolume {i+1}/{volume_points}: {scale_factor:.3f}")

        scaled_atoms = equilibrium_atoms.copy()
        scaled_atoms.set_cell(equilibrium_atoms.get_cell() * scale_factor**(1/3), scale_atoms=True)
        target_cell = scaled_atoms.get_cell().copy()
        vol_target = scaled_atoms.get_volume()
        structure = AseAtomsAdaptor.get_structure(scaled_atoms)

        result = m3gnet_relaxer.relax(structure, **relax_kwargs)
        final_structure = result["final_structure"]
        final_atoms = AseAtomsAdaptor.get_atoms(final_structure)

        vol = final_atoms.get_volume()
        if abs(vol - vol_target) / vol_target > 0.005:
            print(f"  ⚠️ volume drifted {vol_target:.3f} → {vol:.3f} Å³; locking cell to target")
            final_atoms = _lock_volume(final_atoms, target_cell)
            final_structure = AseAtomsAdaptor.get_structure(final_atoms)
            # ions were optimized at wrong V — short re-relax with cell locked
            result = m3gnet_relaxer.relax(final_structure, **relax_kwargs)
            final_structure = result["final_structure"]
            final_atoms = AseAtomsAdaptor.get_atoms(final_structure)
            if abs(final_atoms.get_volume() - vol_target) / vol_target > 0.005:
                final_atoms = _lock_volume(final_atoms, target_cell)
                final_structure = AseAtomsAdaptor.get_structure(final_atoms)

        # energy at the locked volume (not trajectory last step at drifted cell)
        _, final_energy = get_forces_from_potential(m3gnet_relaxer, final_structure)
        vol = final_atoms.get_volume()

        volumes.append(vol)
        energies.append(final_energy)
        print(f"✓ Vol={vol:.2f} Å³ (target {vol_target:.2f}), E={final_energy:.6f}")

        np.savetxt(f"ev_point_{i:02d}.dat", [[vol, final_energy]],
                   header=f"# V={vol:.4f} E={final_energy:.6f}")

        unitcell = PhonopyAtoms(
            symbols=final_atoms.get_chemical_symbols(),
            cell=final_atoms.get_cell(),
            scaled_positions=final_atoms.get_scaled_positions(),
        )
        phonon = Phonopy(unitcell, supercell_matrix=mul_matrix)
        phonon.generate_displacements(distance=0.02)

        supercells = phonon.supercells_with_displacements
        sets_of_forces = []
        print(f"  Forces: {len(supercells)} displacements")

        for j, scell in enumerate(supercells):
            atoms_disp = Atoms(
                symbols=scell.symbols, positions=scell.positions, cell=scell.cell, pbc=True
            )
            struct_disp = AseAtomsAdaptor.get_structure(atoms_disp)
            forces, _ = get_forces_from_potential(m3gnet_relaxer, struct_disp)
            assert forces.shape == (len(scell.symbols), 3)
            sets_of_forces.append(forces)
            if (j + 1) % 3 == 0:
                print(f"    {j+1}/{len(supercells)}")

        phonon.forces = sets_of_forces
        phonon.produce_force_constants()
        phonon.run_mesh([50, 50, 50])
        phonon.run_thermal_properties(t_step=10, t_max=1500, t_min=0)
        phonon.write_yaml_thermal_properties(f"thermal_properties.yaml-{i:02d}")

    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    dv_frac = (volumes.max() - volumes.min()) / volumes.mean()
    print(f"\nEOS span: ΔV/V = {dv_frac:.2%}  V=[{volumes.min():.2f}, {volumes.max():.2f}]")
    if dv_frac < 0.03:
        raise RuntimeError(
            f"EOS volume collapsed (ΔV/V={dv_frac:.2%}). "
            "Relaxer is still changing the cell — ensure Relaxer(..., relax_cell=False)."
        )

    print("\n💾 Saving QHA dataset...")
    np.savetxt("e-v.dat", np.column_stack([volumes, energies]), fmt="%.8f",
               header="# volume(Å³) energy(eV)")

    yaml_files = sorted(glob.glob("thermal_properties.yaml-*"))
    if not yaml_files:
        raise FileNotFoundError("No thermal_properties.yaml-* written; cannot run phonopy-qha")

    phonopy_qha_path = None
    for candidate in (
        f"/home/a.burov/micromamba/envs/{env_used}/bin/phonopy-qha",
        "/home/a.burov/micromamba/envs/msdb/bin/phonopy-qha",
        shutil.which("phonopy-qha"),
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            phonopy_qha_path = candidate
            break
    if phonopy_qha_path is None:
        raise FileNotFoundError(
            f"phonopy-qha not found in {env_used} or msdb. "
            "Install phonopy in m3gnet-env or point env_used at an env that has it."
        )

    qha_env = os.environ.copy()
    qha_env["MPLBACKEND"] = "Agg"
    yaml_list = " ".join(yaml_files)
    cmd = (
        f"{phonopy_qha_path} -s --tmax=1000 --cutoff-frequency 0.1 "
        f"e-v.dat {yaml_list}"
    )
    print(f"\nRunning phonopy-qha from: {phonopy_qha_path}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=qha_env)
    if result.returncode != 0 and (
        "unrecognized arguments" in (result.stderr or "")
        or "invalid option" in (result.stderr or "").lower()
    ):
        print("phonopy-qha does not support --cutoff-frequency; retrying without it")
        cmd = f"{phonopy_qha_path} -s --tmax=1000 e-v.dat {yaml_list}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=qha_env)

    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"phonopy-qha failed:\n{result.stderr or result.stdout}")
    if not os.path.isfile("helmholtz-volume_fitted.dat"):
        raise FileNotFoundError(
            "phonopy-qha exited 0 but did not write helmholtz-volume_fitted.dat.\n"
            f"stderr:\n{result.stderr}"
        )
    print("QHA complete!")

    return volumes, energies



# In[ ]:





# In[ ]:


# In[ ]:





# In[ ]:





# In[ ]:


def visualize_data(potential, structure):
    """Helmholtz / α / G plots from phonopy-qha outputs."""
    fig_dir = _phonon_fig_dir()
    os.makedirs(fig_dir, exist_ok=True)

    qha_files = (
        "helmholtz-volume_fitted.dat",
        "thermal_expansion.dat",
        "gibbs-temperature.dat",
    )
    missing = [f for f in qha_files if not os.path.isfile(f)]
    if missing:
        print(
            "Skipping plots: phonopy-qha did not write "
            + ", ".join(missing)
            + ". Check the QHA log above."
        )
        return

    with open("helmholtz-volume_fitted.dat") as f:
        lines = f.readlines()

    fitted_start = None
    minima_start = None
    for i, line in enumerate(lines):
        if "# Fitted data" in line:
            fitted_start = i + 1
        if "# Minimas" in line:
            minima_start = i + 1
            break

    fitted_data = []
    for line in lines[fitted_start:minima_start - 1]:
        if line.strip() and not line.startswith("#"):
            fitted_data.append([float(x) for x in line.split()])
    fitted_data = np.array(fitted_data)
    volumes_fitted = fitted_data[:, 0]
    helmholtz_fitted = fitted_data[:, 1:]

    minima_data = []
    for line in lines[minima_start:]:
        if line.strip() and not line.startswith("#"):
            minima_data.append([float(x) for x in line.split()])
    minima_data = np.array(minima_data)
    vol_eq = minima_data[:, 0]
    energy_eq = minima_data[:, 1]

    n_temps = helmholtz_fitted.shape[1]
    print(f"Number of temperatures: {n_temps}")
    print(f"Number of volume points: {len(volumes_fitted)}")
    print(f"Equilibrium points: {len(vol_eq)}")

    fig, ax = plt.subplots(figsize=(8, 10))
    N = 5
    for i in range(n_temps):
        ax.plot(volumes_fitted[::N], helmholtz_fitted[::N, i], "o-", color="blue",
                linewidth=1.5, markersize=4, alpha=0.8)
    ax.plot(vol_eq, energy_eq, "*-", color="red", linewidth=1.5, markersize=10, zorder=10)
    ax.set_xlabel("Volume (Å³)", fontsize=fontsize)
    ax.set_ylabel("Free energy (eV)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{potential}_free_{structure}.png", dpi=450)
    plt.savefig(f"{fig_dir}/{potential}_free_{structure}.pdf", dpi=450)
    plt.close()
    print(f"Plot saved to: {fig_dir}/{potential}_free_{structure}")
    print(f"\nEquilibrium volumes:")
    print(f"  At T=0 K:   {vol_eq[0]:.4f} Å³")
    print(f"  At T={n_temps-1}00 K: {vol_eq[-1]:.4f} Å³")
    print(f"  Expansion: {vol_eq[-1] - vol_eq[0]:.4f} Å³")

    data = np.loadtxt("thermal_expansion.dat")
    temperatures, alpha = data[:, 0], data[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temperatures, alpha, "-", color="red", linewidth=2.5)
    ax.set_xlabel("Temperature (K)", fontsize=fontsize)
    ax.set_ylabel(r"Thermal expansion (K$^{-1}$)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(fontsize - 2)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{potential}_thermal_{structure}.pdf", dpi=450)
    plt.savefig(f"{fig_dir}/{potential}_thermal_{structure}.png", dpi=450)
    plt.close()
    print("✓ Thermal expansion coefficient plot saved")
    print(f"\nα at 0 K: {alpha[0]:.2e} ")
    print(f"α at 300 K: {alpha[30]:.2e}")
    print(f"α at 1000 K: {alpha[100]:.2e}")

    data = np.loadtxt("gibbs-temperature.dat")
    temperatures, gibbs = data[:, 0], data[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temperatures, gibbs, "-", color="red", linewidth=2.5)
    ax.set_xlabel("Temperature (K)", fontsize=fontsize)
    ax.set_ylabel("Gibbs free energy (eV)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{potential}_gibbs_{structure}.png", dpi=450)
    plt.savefig(f"{fig_dir}/{potential}_gibbs_{structure}.pdf", dpi=450)
    plt.close()
    print("✓ Gibbs free energy plot saved")
    print(f"\nG at 0 K:    {gibbs[0]:.2f} eV")
    print(f"G at 300 K:  {gibbs[30]:.2f} eV")
    print(f"G at 1000 K: {gibbs[100]:.2f} eV")



# In[ ]:


# In[ ]:





# In[15]:


def save_files(path_save):
    # create destination directory
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    # Define patterns to match
    patterns = [
        "*.pdf",
        "*.dat", 
        "*.png",
        "*.log",
        "*.traj",
        "thermal_properties.yaml*"
    ]
    
    # Move all matching files
    moved_count = 0
    for pattern in patterns: 
        if (pattern != f"{potential}.log"):
            files = glob.glob(pattern)
            for file in files:
                dest_path = os.path.join(path_save, os.path.basename(file))
                
                # Remove destination file if it exists
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                
                shutil.move(file, dest_path)  # Changed: use dest_path instead of path_save
                # print(f"Moved: {file} -> {dest_path}")
                moved_count += 1
    
    print(f"\n Total files moved: {moved_count}")


# In[ ]:





# In[16]:


# Load equilibrium structure
# equilibrium_atoms = read("/home/a.burov/icys_2025/niohf/optimized/dft/delta.cif")


# In[17]:


# equilibrium_atoms = bulk('Al', 'fcc', a=4.05)



# In[18]:


path_base = (
    "/home/arseniy/Desktop/work/niohf/optimized/m3gnet"
    if Path("/home/arseniy/Desktop/work/niohf/optimized/m3gnet").exists()
    else "/home/a.burov/icys_2025/niohf/optimized/m3gnet/"
)
_UMLP_OUT = (
    Path("/home/arseniy/Desktop/work/niohf/umlip_phonons")
    if Path("/home/arseniy/Desktop/work/niohf/umlip_phonons").exists()
    else Path("/home/a.burov/icys_2025/niohf/umlip_phonons")
)


# In[19]:


structures_phases = [
    "layered_p1", 
    "layered_p-1",
    "layered_p21",
    "layered_p2c",
    "layered_p21c",
    
    "alpha", 
    "beta", 
    "gamma", 
    "delta",      
]


# In[ ]:





# In[ ]:





# In[20]:


for st in structures_phases:
    if "layered" in st:
        mul_matrix=[[2,0,0], [0,2,0], [0,0,2]]
        scale = [0.92, 1.14]  # wider than [0.97, 1.10] so QHA well is interior after cell-fixed relax
    else:
        mul_matrix=[[4,0,0], [0,3,0], [0,0,1]]
        scale = [0.92, 1.12]
         
    
    path_file = f"{path_base}/{st}.cif"  
    # read relaxed structures
    equilibrium_atoms = read(path_file)
    # equilibrium_atoms = bulk('Al', 'fcc', a=4.05)

    # path to save phonopy data
    path_data_phonopy = str(_UMLP_OUT / potential / st)

    calc_phonopy(equilibrium_atoms, m3gnet_relaxer, mul_matrix=mul_matrix, env_used=env_used, scale=scale, volume_points=11)

    # visualize data and save plots (no-op if phonopy-qha did not write files)
    visualize_data(potential=potential, structure=st)

    # save phonopy files
    save_files(path_data_phonopy)

    print("Finished without errors!")


    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




