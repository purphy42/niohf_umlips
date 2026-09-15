#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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





# In[ ]:


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


def m3gnet_calc(st, st_name, path_save, max_force=0.05, steps=20):
    """
    INPUT:
        st (ase Structure)
    RETURN:
        optimized structure with same atom ordering
    """
    
    # Store original positions and species
    original_symbols = [site.specie.symbol for site in st]
    
    # Optimize
    result = m3gnet_model.relax(st, verbose=True, fmax=max_force, steps=steps)

    # Relaxed structure
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





# In[ ]:


# path to save data
path_save = "/home/a.burov/icys_2025/niohf/optimized/"


# In[ ]:


# create directory for files if it does not exist
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

# create directories for each type of calculations if they do not exist
Path(f"{path_save}/m3gnet").mkdir(parents=True, exist_ok=True)


# In[ ]:





# In[ ]:


path_save_m3gnet = path_save + "/m3gnet/"


# In[ ]:


m3gnet_model = Relaxer(optimizer='FIRE', relax_cell=True)  # This loads the default pre-trained model


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
    st_relaxed_m3gnet, result = m3gnet_calc(st_pmg.copy(), name, path_save_m3gnet, steps=300)

    data_dir[name]['energy_m3gnet'] = result["trajectory"].energies[-1]



# In[ ]:





# In[ ]:


import csv


# In[ ]:


path_energies = '/home/a.burov/icys_2025/niohf/optimized/energies_m3gnet.csv'


# In[ ]:


# choose keys and order
fieldnames = ['name', 'energy_m3gnet']


rows = []

for key, val in data_dir.items():
    rows.append({
        'key': key,
        'energy_m3gnet': val['energy_m3gnet'],
    })


with open(path_energies, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for key, val in data_dir.items():
        w.writerow([key, val['energy_m3gnet'], ])
        

    


# In[ ]:





# In[ ]:


print(f"CSV written to: {path_energies}")


# In[ ]:





# In[ ]:




