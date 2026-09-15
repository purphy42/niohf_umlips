#!/usr/bin/env python
# coding: utf-8

# ## Other default imports 

# In[10]:


import os, re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
# from pymatgen.ext.matproj import MPRester
from mp_api.client import MPRester
from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.interface_reactions import InterfacialReactivity, GrandPotentialInterfacialReactivity
from emmet.core.thermo import ThermoType
from pymatgen.entries.computed_entries import ComputedEntry
from matplotlib.patches import Patch
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pathlib import Path
from siman.core.structure import Structure as base_siman_structure


# In[1]:


# %%capture
import siman #program package to manage DFT calculations https://github.com/dimonaks/siman
from siman.calc_manage import smart_structure_read, get_structure_from_matproj, get_structure_from_matproj_new
from siman.calc_manage import add, res
# Update configurations
from siman import header
from siman.database import write_database, read_database
from siman.set_functions import read_vasp_sets
from siman.header import db
from siman.header import _update_configuration
_update_configuration('project_conf.py')
read_database() # read saved database if available
from pydoc import importfile
project_sets = importfile('project_sets.py')
varset = read_vasp_sets(project_sets.user_vasp_sets, override_global = 1) #read user sets

from siman import thermo

# header.PATH2PROJECT = '../dft_calculations/'
header.PATH2EDITOR = 'notepad.exe'
header.PATH2POTENTIALS = "/home/a.burov/soft/vasp_potentials/potpaw_PBE_MPIE/"

from matplotlib import rc



# In[12]:


from ase.optimize import FIRE  
from ase.io import read, write
from ase.visualize import view

from ase.constraints import ExpCellFilter


# In[ ]:





# In[13]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[14]:


np.set_printoptions(precision=4)


# In[15]:


# %matplotlib inline
plt.rcParams['figure.dpi'] = 450


# In[ ]:





# In[ ]:





# ## UMLIP calculations

# In[17]:


from sevenn.calculator import SevenNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor



# In[18]:


from chgnet.model import StructOptimizer


# In[ ]:





# In[19]:


data_dir = {
    "layered_p1": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/layered/layered_P1.POSCAR"},
    "layered_p-1": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/layered/layered_P-1.POSCAR"},
    "layered_p21": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/layered/layered_P2_1.POSCAR"},
    "layered_p2c": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/layered/layered_Pc.POSCAR"},
    "layered_p21c": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/layered/layered_P2_1_c.POSCAR"},
    
    "alpha": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_alpha.POSCAR"}, 
    "beta": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_beta.POSCAR"}, 
    "gamma": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_gamma.POSCAR"}, 
    "delta": {"path": "/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_delta.POSCAR"},      
}


# In[ ]:





# In[27]:


def sevenn_calc(st, st_name, path_save, max_force=0.05, steps=20):
    """
    INPUT:
        st (ase Structure)
    RETURN:
        optimized structure with same atom ordering
    """
    
    # Store original positions and species
    original_symbols = st.get_chemical_symbols()
    original_positions = st.get_scaled_positions().copy()
    
    # Assign calculator
    st.calc = sevennet_0_cal
    constraint = ExpCellFilter(st)
    
    # Optimize
    optim = FIRE(constraint, dt=0.05)
    optim.run(fmax=max_force, steps=steps)
    
    # Check if order changed
    new_symbols = st.get_chemical_symbols()
    if new_symbols != original_symbols:
        print(f"WARNING: Atom order changed for {st_name}!")
        
        # Try to restore order by matching positions
        from scipy.optimize import linear_sum_assignment
        
        new_positions = st.get_scaled_positions()
        n_atoms = len(st)
        dist_matrix = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                diff = original_positions[i] - new_positions[j]
                diff = diff - np.round(diff)
                dist_matrix[i, j] = np.linalg.norm(diff)
        
        _, col_ind = linear_sum_assignment(dist_matrix)
        
        # Reorder
        st = st[col_ind]
        print(f"  Reordered atoms for {st_name}")
    
    # Save
    st.write(path_save + f"/{st_name}.cif")
    st.write(path_save + f"/{st_name}.xyz")    
    st.write(path_save + f"/{st_name}.vasp")    
    
    return st, optim


# In[ ]:





# In[21]:


def chgnet_calc(st, st_name, path_save, max_force=0.05, steps=20):
    """
    INPUT:
        st (Pymatgen Structure)
    RETURN:
        relaxed structure (with preserved atom order), result dictionary
    """
    
    # Store original ordering
    original_symbols = [site.specie.symbol for site in st]
    
    # Relax structure
    result = relaxer.relax(st, fmax=max_force, steps=steps)
    st_relaxed = result["final_structure"]
    
    # Check if atom order changed
    relaxed_symbols = [site.specie.symbol for site in st_relaxed]
    
    if relaxed_symbols != original_symbols:
        print(f"WARNING: Atom order changed in {st_name} during CHGNet relaxation!")
        
        # Match atoms based on positions
        from scipy.optimize import linear_sum_assignment
        
        orig_coords = np.array([site.frac_coords for site in st])
        relax_coords = np.array([site.frac_coords for site in st_relaxed])
        
        # Build distance matrix with PBC
        n_atoms = len(st)
        dist_matrix = np.zeros((n_atoms, n_atoms))
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                diff = orig_coords[i] - relax_coords[j]
                diff = diff - np.round(diff)  # PBC
                cart_diff = st.lattice.get_cartesian_coords(diff)
                dist_matrix[i, j] = np.linalg.norm(cart_diff)
        
        # Find optimal matching
        _, col_ind = linear_sum_assignment(dist_matrix)
        
        # Reorder relaxed structure
        sites_reordered = [st_relaxed[j] for j in col_ind]
        from pymatgen.core import Structure
        st_relaxed = Structure.from_sites(sites_reordered)
        
        print(f"  Reordered atoms for {st_name}")
    
    # Save
    st_relaxed.to(filename=path_save + f"/{st_name}.cif")

    atoms_relaxed = AseAtomsAdaptor.get_atoms(st_relaxed)  # convert to ASE Atoms [web:68]
    write(path_save + f'/{st_name}.xyz', atoms_relaxed)     # ASE infers format from .xyz [web:62]
    write(path_save + f'/{st_name}.vasp', atoms_relaxed)     # ASE infers format from .xyz [web:62]
    
    return st_relaxed, result



# In[ ]:





# In[6]:


# path to save data
path_save = "/home/a.burov/icys_2025/niohf/optimized/"


# In[7]:


# create directory for files if it does not exist
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

# create directories for each type of calculations if they do not exist
Path(f"{path_save}/dft").mkdir(parents=True, exist_ok=True)
Path(f"{path_save}/sevenn").mkdir(parents=True, exist_ok=True)
Path(f"{path_save}/chgnet").mkdir(parents=True, exist_ok=True)


# In[ ]:





# In[22]:


path_save_sevenn = path_save + "/sevenn/"
path_save_chgnet = path_save + "/chgnet"


# In[25]:


# create CHGNet calculator
relaxer = StructOptimizer(use_device='cpu', optimizer_class="FIRE")

# create SevenNet calculator
sevennet_0_cal = SevenNetCalculator(model="7net-mf-ompa", device='cpu', modal="mpa") 


# In[ ]:





# In[29]:


for name, vals in data_dir.items():
    path_file = vals['path']

    if name == "layered":
        st_siman = smart_structure_read(path_file)
        els = st_siman.get_elements()
        els_ni = [idx for idx, el in enumerate(els) if el == "Cu"]
        st_siman = st_siman.replace_atoms(els_ni, "Ni")
        st_pmg = st_siman.convert2pymatgen()
        # en_dft = db[name, '9bulk', 1].e0
    else:
        st_pmg = Structure.from_file(path_file)
        # en_dft = db[f"{name}.sc", '9bulk_eos', 100].e0
    
    # Store original ordering information
    st_pmg.add_site_property('original_index', list(range(len(st_pmg))))

    # Convert to ASE
    st_ase = AseAtomsAdaptor.get_atoms(st_pmg)
    
    # Store original indices as tags
    st_ase.set_tags(np.arange(len(st_ase)))

    st_ase.write(path_save + f"/dft/{name}.xyz")    
    st_ase.write(path_save + f"dft/{name}.cif")    
    st_ase.write(path_save + f"dft/{name}.vasp")    
    
    # Run calculations
    st_relaxed_sevenn, _ = sevenn_calc(st_ase.copy(), name, path_save + "/sevenn", steps=300)
    
    st_pmg_copy = st_pmg.copy()
    st_relaxed_chgnet, result = chgnet_calc(st_pmg_copy, name, path_save + "/chgnet", steps=300)

    # data_dir[name]['energy_dft'] = en_dft
    data_dir[name]['energy_sevennet'] = st_relaxed_sevenn.get_potential_energy()
    data_dir[name]['energy_chgnet'] = result["trajectory"].energies[-1]
    


# In[ ]:





# In[ ]:





# In[30]:


import csv


# In[31]:


path_energies = '/home/a.burov/icys_2025/niohf/optimized/energies_sevennet_chgnet.csv'


# In[ ]:





# In[34]:


# choose keys and order
# fieldnames = ['name', 'energy_dft', 'energy_sevennet', 'energy_chgnet']
fieldnames = ['name', 'energy_sevennet', 'energy_chgnet']

rows = []

for key, val in data_dir.items():
    rows.append({
        'key': key,
        'energy_sevennet': val['energy_sevennet'],
        'energy_chgnet': val['energy_chgnet'],
    })


with open(path_energies, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for key, val in data_dir.items():
        w.writerow([key, val['energy_sevennet'], val['energy_chgnet']])
        

    


# In[ ]:





# In[ ]:


print(f"CSV written to: {path_energies}")


# In[ ]:





# In[ ]:




