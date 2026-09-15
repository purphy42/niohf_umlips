#!/usr/bin/env python
# coding: utf-8

# In[1]:


from alignn.ff.ff import AlignnAtomwiseCalculator,default_path
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.atoms import ase_to_atoms


# In[2]:


from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase.io import read, write
from ase.constraints import ExpCellFilter
from ase.optimize.fire import FIRE
import numpy as np


# In[ ]:





# In[3]:


def general_relaxer(ase_atoms="", calculator="", fmax=0.05, steps=0, relax=True, max_step=0.1):
    ase_atoms.calc = calculator
    if not relax:
         return ase_atoms.get_potential_energy()
        
    atoms_constrained = ExpCellFilter(ase_atoms)
    dyn = FIRE(atoms_constrained, maxstep=max_step, dt=0.1)
    dyn.run(fmax=fmax, steps=steps)
    
    return ase_atoms, dyn



# In[ ]:





# In[4]:


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





# In[5]:


def alignn_calc(st, st_name, path_save, max_force=0.05, steps=20):
    """
    INPUT:
        st (ase Structure)
    RETURN:
        optimized structure with same atom ordering
    """

    # Store original positions and species
    original_symbols = st.get_chemical_symbols()
    original_positions = st.get_scaled_positions().copy()
    
    # Assign calculator and optimize 
    st, optim = general_relaxer(ase_atoms=st, calculator=alignn_model, fmax=max_force, steps=steps, relax=True, max_step=0.1)
    
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





# In[ ]:


# path to save data
path_save = "/home/a.burov/icys_2025/niohf/optimized/"


# In[ ]:


# create directory for files if it does not exist
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

# create directories for each type of calculations if they do not exist
Path(f"{path_save}/alignn").mkdir(parents=True, exist_ok=True)


# In[ ]:





# In[ ]:


# download default pre-trained model
# model_path = default_path()


# In[ ]:





# In[ ]:


# path to the model
path_save_alignn= path_save + "/alignn/"


# In[ ]:


model_path = "/home/a.burov/umlip/alignn/v12.2.2024_mp_1.5mill/"


# In[ ]:


alignn_model = AlignnAtomwiseCalculator(path=model_path, model_filename="best_model.pt")



# In[ ]:





# In[ ]:


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
    st_relaxed_alignn, optim = alignn_calc(st_ase.copy(), name, path_save + "/alignn", steps=1200)
    
    data_dir[name]['energy_alignn'] = float(st_relaxed_alignn.get_potential_energy())
    


# In[ ]:





# In[ ]:





# In[ ]:


import csv


# In[ ]:


path_energies = '/home/a.burov/icys_2025/niohf/optimized/energies_alignn.csv'


# In[ ]:





# In[ ]:


# choose keys and order
fieldnames = ['name', 'energy_alignn']

rows = []

for key, val in data_dir.items():
    rows.append({
        'key': key,
        'energy_alignn': val['energy_alignn'],
    })


with open(path_energies, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for key, val in data_dir.items():
        w.writerow([key, val['energy_alignn'], ])
        

    


# In[ ]:





# In[ ]:


print(f"CSV written to: {path_energies}")


# In[ ]:





# In[ ]:




