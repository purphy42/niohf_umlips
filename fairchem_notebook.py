#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fairchem.core.calculate.pretrained_mlip import load_predict_unit
from fairchem.core import FAIRChemCalculator
from ase.io import read


# In[2]:


pot_path = "/home/a.burov/umlip/fairchem/uma-s-1.pt"


# In[ ]:





# In[4]:


from ase.optimize import FIRE  
from ase import Atoms
from ase.optimize import BFGS
from ase import Atoms
from ase.geometry import cellpar_to_cell

from pymatgen.core import Lattice, Structure
from pymatgen.core import Structure
from pathlib import Path
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np



# In[ ]:





# In[5]:


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





# In[ ]:


def fairchem_calc(st, st_name, path_save, max_force=0.05, steps=20):
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
    st.calc = fairchem_model
    
    # Optimize
    optim = FIRE(st, dt=0.05, maxstep=0.5)
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





# In[ ]:


# path to save data
path_save = "/home/a.burov/icys_2025/niohf/optimized/"


# In[ ]:


# create directory for files if it does not exist
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

# create directories for each type of calculations if they do not exist
Path(f"{path_save}/fairchem").mkdir(parents=True, exist_ok=True)


# In[ ]:





# In[ ]:


path_save_fairchem = path_save + "/fairchem/"


# In[ ]:


predict_unit = load_predict_unit(pot_path, device='cpu')


# In[ ]:


fairchem_model = calc = FAIRChemCalculator(predict_unit, task_name="omat")


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
    st_relaxed_fairchem, optim = fairchem_calc(st_ase.copy(), name, path_save + "/fairchem", steps=200)
    
    data_dir[name]['energy_fairchem'] = st_relaxed_fairchem.get_potential_energy()
    


# In[ ]:





# In[ ]:


import csv


# In[ ]:


path_energies = '/home/a.burov/icys_2025/niohf/optimized/energies_fairchem.csv'


# In[ ]:


# choose keys and order
fieldnames = ['name', 'energy_fairchem']


rows = []

for key, val in data_dir.items():
    rows.append({
        'key': key,
        'energy_fairchem': val['energy_fairchem'],
    })


with open(path_energies, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for key, val in data_dir.items():
        w.writerow([key, val['energy_fairchem'], ])
        

    


# In[ ]:





# In[ ]:


print(f"CSV written to: {path_energies}")


# In[ ]:





# In[ ]:




