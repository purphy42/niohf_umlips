#!/usr/bin/env python
# coding: utf-8

# ### Mattersim

# In[43]:


import torch
from ase.build import bulk
from ase.units import GPa
from mattersim.forcefield import DeepCalculator, MatterSimCalculator, Potential
from mattersim.applications.relax import Relaxer

from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
import numpy as np


# In[ ]:





# In[45]:


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





# In[72]:


def mattersim_calc(st, st_name, path_save, max_force=0.05, steps=20, max_step=0.2):
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
    st.calc = mattersim_model
    
    # Optimize
    relaxer = Relaxer(
                    optimizer="FIRE",           # or "FIRE"
                    filter="ExpCellFilter",     # allow cell relaxation
                    constrain_symmetry=True,    # preserve symmetry
            )
    
    relaxed_structure = relaxer.relax(st, steps=steps, fmax=max_force, )
    
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
    
    return st, relaxed_structure


# In[ ]:





# In[57]:


# path to save data
path_save = "/home/a.burov/icys_2025/niohf/optimized/"


# In[ ]:





# In[58]:


# create directory for files if it does not exist
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

# create directories for each type of calculations if they do not exist
Path(f"{path_save}/mattersim").mkdir(parents=True, exist_ok=True)


# In[ ]:





# In[39]:


path_save_mattersim = path_save + "/mattersim/"


# In[40]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running MatterSim on {device}")


# In[48]:


mattersim_model = MatterSimCalculator(load_path="/home/a.burov/umlip/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth",
                                      device=device)


# In[ ]:





# In[73]:


for name, vals in data_dir.items():
    path_file = vals['path']

    st_pmg = Structure.from_file(path_file)
    st_pmg.replace_species({"Cu": "Ni"})
    
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
    st_relaxed_mattersim, optim = mattersim_calc(st_ase.copy(), name, path_save + "/mattersim", steps=200)
    
    data_dir[name]['energy_mattersim'] = optim[1].get_potential_energy()
    


# In[ ]:





# In[ ]:


# print(f"Energy (eV)                 = {relaxed_structure[1].get_potential_energy()}")
# print(f"Energy per atom (eV/atom)   = {relaxed_structure[1].get_potential_energy()/len(si)}")
# print(f"Forces of first atom (eV/A) = {relaxed_structure[1].get_forces()[0]}")
# print(f"Stress[0][0] (eV/A^3)       = {relaxed_structure[1].get_stress(voigt=False)[0][0]}")
# print(f"Stress[0][0] (GPa)          = {relaxed_structure[1].get_stress(voigt=False)[0][0] / GPa}")


# In[74]:


import csv


# In[ ]:





# In[75]:


path_energies = '/home/a.burov/icys_2025/niohf/optimized/energies_mattersim.csv'


# In[ ]:





# In[76]:


# choose keys and order
fieldnames = ['name', 'energy_mattersim']

rows = []

for key, val in data_dir.items():
    rows.append({
        'key': key,
        'energy_mattersim': val['energy_mattersim'],
    })


with open(path_energies, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for key, val in data_dir.items():
        w.writerow([key, val['energy_mattersim'], ])
        

    


# In[ ]:





# In[77]:


print(f"CSV written to: {path_energies}")


# In[ ]:





# In[ ]:




