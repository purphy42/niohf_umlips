#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Internal phonopy methods

# In[1]:


import torch
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import DeepCalculator, MatterSimCalculator, Potential
from mattersim.applications.relax import Relaxer

from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
import numpy as np


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.optimize import FIRE
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import yaml
import subprocess

from ase.calculators.eam import EAM

import os
import shutil
import glob
from pathlib import Path

from ase.build import bulk
from ase.constraints import ExpCellFilter

import pandas as pd


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





# In[9]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running MatterSim on {device}")



# In[10]:


mattersim_cal = MatterSimCalculator(load_path="/home/a.burov/umlip/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
                                      device=device)




# In[ ]:





# In[11]:


fontsize = 20


# In[12]:


env_used = "mattersim-env"


# In[13]:


potential = "mattersim"


# In[ ]:





# In[14]:


def calc_phonopy(equilibrium_atoms, mul_matrix=[[2,0,0], [0,2,0], [0,0,2]], env_used="msdb", scale=[0.97, 1.10] ):

    # Volume scaling
    volume_points = 6
    scale_init = scale[0]
    scale_end = scale[1]
    volume_scales = np.linspace(scale_init, scale_end, volume_points)
    volumes = []
    energies = []
    
    print("Starting QHA calculation with mattersim...")
    
    for i, scale in enumerate(volume_scales):
        print(f"\n{'='*60}")
        print(f"Volume point {i+1}/{volume_points}: scale = {scale:.4f}")
        print(f"{'='*60}")
        
        # Scale structure
        scaled_atoms = equilibrium_atoms.copy()
        scaled_atoms.set_cell(equilibrium_atoms.get_cell() * scale**(1/3), scale_atoms=True)
        
        # Relax at fixed volume
        scaled_atoms.calc = mattersim_cal
        opt = FIRE(scaled_atoms, trajectory=f"vol_{i:02d}_relax.traj", logfile=f"vol_{i:02d}_opt.log", dt=0.05)
        opt.run(fmax=1e-4, steps=500)
        
        vol = scaled_atoms.get_volume()
        en = scaled_atoms.get_potential_energy()
        volumes.append(vol)
        energies.append(en)
        
        print(f"Volume: {vol:.4f} Å³, Energy: {en:.6f} eV")
        
        # Convert to Phonopy
        unitcell = PhonopyAtoms(
            symbols=scaled_atoms.get_chemical_symbols(),
            cell=scaled_atoms.get_cell(),
            scaled_positions=scaled_atoms.get_scaled_positions()
        )
        
        # Create Phonopy object
        phonon = Phonopy(unitcell, supercell_matrix=mul_matrix)  # Changed: use mul_matrix parameter
        phonon.generate_displacements(distance=0.02)

        # Get supercell information
        supercell = phonon.supercell
        sc_lattice = supercell.cell  # Changed: removed _params
        sc_a = np.linalg.norm(sc_lattice[0])
        sc_b = np.linalg.norm(sc_lattice[1])
        sc_c = np.linalg.norm(sc_lattice[2])
        n_atoms_super = len(supercell)
        
        print(f"Supercell lattice: a={sc_a:.3f}, b={sc_b:.3f}, c={sc_c:.3f} Å")
        print(f"Supercell atoms: {n_atoms_super}")
        print(f"Supercell volume: {supercell.volume:.4f} Å³")  # Added for info
        
        # Calculate forces
        supercells = phonon.supercells_with_displacements
        sets_of_forces = []
        
        print(f"Calculating {len(supercells)} displaced structures...")
        for j, scell in enumerate(supercells):
            atoms_disp = Atoms(
                symbols=scell.symbols,
                positions=scell.positions,
                cell=scell.cell,
                pbc=True
            )
            atoms_disp.calc = mattersim_cal
            forces = atoms_disp.get_forces()
            sets_of_forces.append(forces)
        
        # Set forces and produce force constants
        phonon.forces = sets_of_forces
        phonon.produce_force_constants()
        
        # Calculate thermal properties
        phonon.run_mesh([50, 50, 50])
        phonon.run_thermal_properties(t_step=10, t_max=1500, t_min=0)
        phonon.write_yaml_thermal_properties(filename=f"thermal_properties.yaml-{i:02d}")
        print(f"Saved thermal_properties.yaml-{i:02d}")
    
    np.savetxt("e-v.dat", np.column_stack([volumes, energies]), 
               fmt="%.8f", header="# volume(Å³)  energy(eV)")
    print("\nSaved e-v.dat")
    
    # Use the environment directly
    phonopy_qha_path = f'/home/a.burov/micromamba/envs/{env_used}/bin/phonopy-qha'
    
    print(f"\nRunning phonopy-qha from: {phonopy_qha_path}")
    
    result = subprocess.run(
        f"{phonopy_qha_path} -p -s --tmax=1000 --cutoff-frequency 0.1 e-v.dat thermal_properties.yaml-*",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("QHA complete!")
        print(result.stdout)
    else:
        print("QHA failed:")
        print(result.stderr)



# In[ ]:





# In[ ]:





# In[15]:


def visualize_data(potential, structure):
    """Helmholtz / α / G plots from phonopy-qha outputs."""
    fig_dir = "/home/a.burov/icys_2025/niohf/data/figures/phonons"
    os.makedirs(fig_dir, exist_ok=True)

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





# In[16]:


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
    
    print(f"\n✓ Total files moved: {moved_count}")


# In[ ]:





# In[17]:


path_base = "/home/a.burov/icys_2025/niohf/optimized/mattersim/"


# In[18]:


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


for st in structures_phases:
    if "layered" in st:
        mul_matrix=[[2,0,0], [0,2,0], [0,0,2]]
        scale = [0.97, 1.10]
    else:
        mul_matrix=[[4,0,0], [0,3,0], [0,0,1]]
        scale = [0.97, 1.08]
    
    path_file = f"{path_base}/{st}.cif"  
    # read relaxed structures
    equilibrium_atoms = read(path_file)

    # path to save phonopy data
    path_data_phonopy = f"/home/a.burov/icys_2025/niohf/umlip_phonons/{potential}/{st}"

    # perform phonopy calculations
    calc_phonopy(equilibrium_atoms, env_used=env_used, mul_matrix=mul_matrix, scale=scale )

    # visualize data and save plots
    visualize_data(potential=potential, structure=st)

    # save phonopy files
    save_files(path_data_phonopy)

    print("Finished without errors!")


    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




