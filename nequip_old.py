#!/usr/bin/env python
# coding: utf-8

# ## NequIP

# In[1]:


from ase.io import read, write
from ase.optimize import FIRE
from tqdm import tqdm

from ase import Atoms
from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt
from nequip.ase import NequIPCalculator


# In[ ]:


from ase.optimize import FIRE  
from ase import Atoms
from ase.optimize import BFGS
from ase import Atoms
from ase.geometry import cellpar_to_cell


# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


calculator = NequIPCalculator.from_compiled_model(
    compile_path="/home/a.burov/umlip/nequip/mir-group__NequIP-OAM-L__0.1.nequip.pt2",  # ← the folder with model/
    device="cpu",
    # chemical_species_to_atom_type_map = {"Ni": "Ni", "O": "O", "H": "H", "F": "F"},
    chemical_species_to_atom_type_map = {"Si": "Si",},
)

# model_path="/home/a.burov/umlip/nequip/mir-group__NequIP-OAM-XL__0.1.nequip.pt2",  # ← the folder with model/


# In[ ]:





# In[ ]:


# Range of scaling factors for lattice constant
scaling_factors = np.linspace(0.95, 1.15, 7)
volumes = []
energies = []

# Loop through scaling factors, calculate energy, and collect volumes and energies
for scale in tqdm(scaling_factors):

    # Generate the cubic silicon structure with 216 atoms
    scaled_si = bulk("Si", crystalstructure="diamond", a=5.43 * scale, cubic=True)
    scaled_si *= (3, 3, 3)  # Make a supercell (3x3x3) to get 216 atoms
    scaled_si.calc = calculator

    volume = scaled_si.get_volume()
    energy = scaled_si.get_potential_energy()
    volumes.append(volume)
    energies.append(energy)


# In[ ]:





# In[ ]:


# Plot the energy-volume curve
plt.figure(figsize=(8, 6))
plt.plot(volumes, energies, marker='o', label="E-V Curve")
plt.xlabel("Volume (Å³)", fontsize=14)
plt.ylabel("Energy (eV)", fontsize=14)
plt.title("Energy-Volume Curve for Cubic Silicon", fontsize=16)
plt.legend(fontsize=12)
plt.grid()
plt.show()


# In[ ]:





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


def nequip_calc(st, st_name, path_save, max_force=0.05, steps=20):
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
    st.calc = nequip_model
    
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





# In[ ]:


# create directory for files if it does not exist
Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

# create directories for each type of calculations if they do not exist
Path(f"{path_save}/nequip").mkdir(parents=True, exist_ok=True)


# In[ ]:





# In[ ]:


path_save_nequip = path_save + "/nequip/"


# In[ ]:





# In[ ]:


nequip_model = nequip_mp(model="large", device="cpu", default_dtype="float64", dispersion=False)


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
    st_relaxed_nequip, optim = nequip_calc(st_ase.copy(), name, path_save + "/nequip", steps=200)
    
    data_dir[name]['energy_nequip'] = st_relaxed_nequip.get_potential_energy()
    


# In[ ]:





# In[ ]:


import csv


# In[ ]:





# In[ ]:


path_energies = '/home/a.burov/icys_2025/niohf/optimized/energies_nequip.csv'


# In[ ]:





# In[ ]:


# choose keys and order
fieldnames = ['name', 'energy_nequip']


rows = []

for key, val in data_dir.items():
    rows.append({
        'key': key,
        'energy_nequip': val['energy_nequip'],
    })


with open(path_energies, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for key, val in data_dir.items():
        w.writerow([key, val['energy_nequip'], ])
        

    


# In[ ]:




