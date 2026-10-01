#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



# In[2]:


MODEL_DIR = "/home/a.burov/soft/umlip/potentials"
from prophet import KairosCalculator


# In[ ]:





# In[3]:


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





# In[4]:


def prophet_calc(st, st_name, path_save, max_force=0.05, steps=20):
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
    st.calc = prophet_model
    
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





# In[5]:


# path to save data
path_save = "/home/a.burov/icys_2025/niohf/optimized/"


# In[6]:


# create directory for files if it does not exist
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

# create directories for each type of calculations if they do not exist
Path(f"{path_save}/prophet").mkdir(parents=True, exist_ok=True)


# In[ ]:





# In[7]:


path_save_prophet = path_save + "/prophet/"


# In[8]:


prophet_model = KairosCalculator(
    model_path=f"{MODEL_DIR}/prophet-oame-mbd.pt",
    use_kernel=False,  # CPU path; zen4 has no CUDA toolchain
    device="cpu",
)


# In[ ]:





# In[9]:


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
    st_relaxed_prophet, optim = prophet_calc(st_ase.copy(), name, path_save + "/prophet", steps=300)
    
    data_dir[name]['energy_prophet'] = st_relaxed_prophet.get_potential_energy()
    


# In[ ]:





# In[ ]:





# In[17]:


import csv


# In[18]:


path_energies = '/home/a.burov/icys_2025/niohf/optimized/energies_prophet.csv'


# In[19]:


# choose keys and order
fieldnames = ['name', 'energy_prophet']


rows = []

for key, val in data_dir.items():
    rows.append({
        'key': key,
        'energy_prophet': val['energy_prophet'],
    })


with open(path_energies, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for key, val in data_dir.items():
        w.writerow([key, val['energy_prophet'], ])
        

    


# In[ ]:





# In[20]:


print(f"CSV written to: {path_energies}")


# In[ ]:





# In[ ]:




