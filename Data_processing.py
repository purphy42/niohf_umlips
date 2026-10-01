#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ase.optimize import FIRE  
import numpy as np
from pathlib import Path
import csv


# ## Other default imports 

# In[2]:


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



# In[3]:


get_ipython().run_cell_magic('capture', '', 'import siman #program package to manage DFT calculations https://github.com/dimonaks/siman\nfrom siman.calc_manage import smart_structure_read, get_structure_from_matproj, get_structure_from_matproj_new\nfrom siman.calc_manage import add, res\n# Update configurations\nfrom siman import header\nfrom siman.database import write_database, read_database\nfrom siman.set_functions import read_vasp_sets\nfrom siman.header import db\nfrom siman.header import _update_configuration\n_update_configuration(\'project_conf.py\')\nread_database() # read saved database if available\nfrom pydoc import importfile\nproject_sets = importfile(\'project_sets.py\')\nvarset = read_vasp_sets(project_sets.user_vasp_sets, override_global = 1) #read user sets\n\nfrom siman import thermo\n\n# header.PATH2PROJECT = \'../dft_calculations/\'\nheader.PATH2EDITOR = \'notepad.exe\'\nheader.PATH2POTENTIALS = "/home/a.burov/soft/vasp_potentials/potpaw_PBE_MPIE/"\n\nfrom matplotlib import rc\n\n')


# In[4]:


from ase.optimize import FIRE  
from ase.io import read, write
from ase.visualize import view


# In[5]:


from matplotlib import font_manager as fm
font_path = "/home/a.burov/fonts/ARIAL.TTF"
fm.fontManager.addfont(font_path)


# In[ ]:





# In[6]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[7]:


np.set_printoptions(precision=5)


# In[8]:


# %matplotlib inline
plt.rcParams['figure.dpi'] = 450


# In[9]:


API_KEY = "LTSM6dStBrl69FjopxP7KdZBP35B1yh7"


# In[ ]:





# In[ ]:





# In[ ]:





# ## DFT Calculations

# ### Layered

# In[10]:


def generate_substitutions(struct):
    """Generate all F ↔ O-H exchange configurations"""
    configs = []
    
    # Identify anion sites (O=8-11, F=12-15)
    O_sites = [8, 9, 10, 11]
    F_sites = [12, 13, 14, 15]
    
    from itertools import combinations
    
    # All possible exchanges (1:1 substitution)
    for n_sub in range(1, 5):  # 1 to 4 exchanges
        for f_idx in combinations(F_sites, n_sub):
            for o_idx in combinations(O_sites, n_sub):
                new_struct = struct.copy()
                
                # Exchange: F → O, O → F at selected sites
                for i, j in zip(f_idx, o_idx):
                    new_struct.replace(i, "O")   # F site → O
                    new_struct.replace(j, "F")   # O site → F
                
                # Check uniqueness (avoid duplicates)
                new_struct.get_sorted_structure()
                configs.append(new_struct)
    
    return configs



# In[11]:


st_layered = smart_structure_read("/home/a.burov/icys_2025/niohf/initial_no_relaxation/CuOHF.cif")


# In[12]:


st_layered.printme()


# In[13]:


st_pmg = st_layered.convert2pymatgen()


# In[14]:


new_configs = generate_substitutions(st_pmg)
print(f"Generated {len(new_configs)} configurations")


# In[ ]:





# In[15]:


from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# In[16]:


def aggressive_filter(configs):
    """Reduce 69 → 5-10 unique structures (fixed API)"""
    unique = []
    
    for config in configs:
        # 1. Niggli reduction
        reduced = config.get_reduced_structure(reduction_algo="niggli")
        
        # 2. Skip if already in unique list
        is_duplicate = False
        for u in unique:
            u_reduced = u.get_reduced_structure(reduction_algo="niggli")
            
            # Fixed: rmsd_cutoff instead of tolerance
            matcher = StructureMatcher(
                primitive_cell=True, 
                # scale=True,
                scale=False, 
                ltol=0.3,  
                stol=0.5,
                angle_tol=5,
                attempt_supercell=False
            )
            if matcher.fit(reduced, u_reduced):
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(config)
    
    return unique


# In[17]:


def fast_symmetry_filter(configs):
    """Ultra-fast symmetry reduction using spacegroup"""
    seen = {}
    unique = []
    
    for i, config in enumerate(configs):
        sg = SpacegroupAnalyzer(config)
        sg_key = sg.get_space_group_symbol()
        formula = config.formula
        
        key = f"{formula}_{sg_key}"
        
        if key not in seen:
            seen[key] = True
            unique.append(config)
            print(f"Added: {formula} | {sg_key}")
    
    return unique


# In[18]:


unique_configs = fast_symmetry_filter(new_configs)
print(f"Unique by symmetry: {len(unique_configs)}")


# In[ ]:





# In[19]:


from siman.core.structure import Structure as base_siman_structure


# In[ ]:





# In[20]:


for st_refined in unique_configs:
    sp_gr = st_refined.get_symmetry_dataset()['international']
    sp_gr = sp_gr.replace("/", "_")
    st_siman = base_siman_structure().update_from_pymatgen(stpm=st_refined)

    els = st_siman.get_elements()
    els_ni = [idx for idx, el in enumerate(els) if el == "Cu"]
    st_siman = st_siman.replace_atoms(els_ni, "Ni")

    print(st_siman.get_formula(), sp_gr)

    if 0:
        st_siman.write_poscar(f"/home/a.burov/icys_2025/niohf/initial_no_relaxation/layered/layered_{sp_gr}.POSCAR")
        st_siman.write_cif(f"/home/a.burov/icys_2025/niohf/initial_no_relaxation/layered/layered_{sp_gr}")

    # add(f"layered_{sp_gr}", '9bulk_eos', 1, it_folder = 'layered', input_st = st_siman, calc_method = 'c_scale', 
				# n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)
    # add(f"layered_{sp_gr}", '9bulk_eos', 1, it_folder = 'layered', input_st = st_siman, calc_method = 'uniform_scale', 
				# n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)
    # add(f"layered_{sp_gr}", '9bulk', 1, it_folder = 'layered', input_st = st_siman, cluster = 'razor128', up='up2', run=2)

    # res(f"layered_{sp_gr}.su", '9bulk_eos', list(range(1,11)) + [100], show = 'fit', analys_type = 'fit_a', up="up2", cluster = 'razor128')
    # res(f"layered_{sp_gr}.sc", '9bulk_eos', list(range(1,11)) + [100], show = 'fit', analys_type = 'fit_a', up="up2", cluster = 'razor128')
    # res(f"layered_{sp_gr}", '9bulk', 1, cluster = 'razor128', )

    st = db[f"layered_{sp_gr}", '9bulk', 1].copy().end
    # add(f"layered_{sp_gr}_rel", '9bulk_eos', 1, it_folder = 'layered', input_st = st, calc_method = 'c_scale', 
				# n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)
    # add(f"layered_{sp_gr}_rel", '9bulk_eos', 1, it_folder = 'layered', input_st = st, calc_method = 'uniform_scale', 
				# n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)
    
    # res(f"layered_{sp_gr}_rel.sc", '9bulk_eos', list(range(1,11)) + [100], show = 'fit', analys_type = 'fit_a', up="up2", cluster = 'razor128')
    # res(f"layered_{sp_gr}_rel.su", '9bulk_eos', list(range(1,11)) + [100], show = 'fit', analys_type = 'fit_a', up="up2", cluster = 'razor128')

    st = db[f"layered_{sp_gr}_rel.sc", '9bulk_eos', 100].copy().end
    # add(f"layered_{sp_gr}_final", '9bulk', 1, it_folder = 'layered', input_st = st, cluster = 'razor128', up='up2', run=2)
    # res(f"layered_{sp_gr}_final", '9bulk', 1, cluster = 'razor128', up='up2', )


    # check calcs
    # add(f"layered_{sp_gr}_final", '9bulk_fine', 1, it_folder = 'layered', input_st = st, cluster = 'razor128', up='up2', run=2)
    # res(f"layered_{sp_gr}_final", '9bulk_fine', 1, cluster = 'razor128', up='up2', )

    # add(f"layered_{sp_gr}_final_su", '9bulk_fine_vdw', 1, it_folder = 'layered', input_st = st, cluster = 'razor128', up='up2', run=2)
    # res(f"layered_{sp_gr}_final_su", '9bulk_fine_vdw', 1, cluster = 'razor128', up='up2', )

    # add(f"layered_{sp_gr}_final_su", '9bulk_fine_rel_vdw', 1, it_folder = 'layered', input_st = st, cluster = 'razor128', up='up2', run=2)
    # res(f"layered_{sp_gr}_final_su", '9bulk_fine_rel_vdw', 1, cluster = 'razor128', up='up2', )
    
    st = db[f"layered_{sp_gr}_rel.su", '9bulk_eos', 100].copy().end
    # add(f"layered_{sp_gr}_final_su", '9bulk', 1, it_folder = 'layered', input_st = st, cluster = 'razor128', up='up2', run=2)
    # res(f"layered_{sp_gr}_final_su", '9bulk', 1, cluster = 'razor128', up='up2', )

    # add(f"layered_{sp_gr}_final_su", '9bulk_fine', 1, it_folder = 'layered', input_st = st, cluster = 'razor128', up='up2', run=2)
    # res(f"layered_{sp_gr}_final_su", '9bulk_fine', 1, cluster = 'razor128', up='up2', )

    # add(f"layered_{sp_gr}_final_su", '9bulk_fine_rel', 1, it_folder = 'layered', input_st = st, cluster = 'razor128', up='up2', run=2)
    # res(f"layered_{sp_gr}_final_su", '9bulk_fine_rel', 1, cluster = 'razor128', up='up2', )


# In[ ]:





# In[21]:


write_database()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[22]:


els = st_layered.get_elements()
els_ni = [idx for idx, el in enumerate(els) if el == "Cu"]


# In[23]:


st_layered = st_layered.replace_atoms(els_ni, "Ni")


# In[24]:


# add("layered", '9bulk_eos', 1, it_folder = 'layered', input_st = st_layered, calc_method = 'c_scale', 
# 				n_scale_images=10, scale_region = (-5, 5), cluster = 'razor128', up='up2', run=2)

# res("layered.sc", '9bulk_eos', list(range(1,11)) + [100], show = 'fit', analys_type = 'fit_a', 
#         up="up2", cluster = 'razor128')


# In[ ]:





# In[25]:


st_rel = db["layered.sc", '9bulk_eos', 100].copy().end


# In[26]:


# add("layered", '9bulk', 1, it_folder = 'layered', input_st = st_rel, cluster = 'razor128', up='up2', run=2)


# In[27]:


# res("layered", '9bulk', 1, cluster = 'razor128', )


# In[ ]:





# In[ ]:





# In[ ]:





# ### Diaspore

# In[28]:


st_alpha = smart_structure_read("/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_alpha.POSCAR")
st_beta = smart_structure_read("/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_beta.POSCAR")
st_gamma = smart_structure_read("/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_gamma.POSCAR")
st_delta = smart_structure_read("/home/a.burov/icys_2025/niohf/initial_no_relaxation/diaspore_NiOHF_delta.POSCAR")


# In[29]:


# add("alpha", '9bulk_eos', 1, it_folder = 'diaspore', input_st = st_alpha, calc_method = 'c_scale', 
# 				n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)


# In[30]:


# add("beta", '9bulk_eos', 1, it_folder = 'diaspore', input_st = st_beta, calc_method = 'c_scale', 
# 				n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)


# In[31]:


# add("gamma", '9bulk_eos', 1, it_folder = 'diaspore', input_st = st_gamma, calc_method = 'c_scale', 
# 				n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)


# In[32]:


# add("delta", '9bulk_eos', 1, it_folder = 'diaspore', input_st = st_delta, calc_method = 'c_scale', 
# 				n_scale_images=10, scale_region = (-10, 10), cluster = 'razor128', up='up2', run=2)


# In[ ]:





# In[ ]:





# In[33]:


for st_name in ["alpha", "beta", "gamma", "delta"]:
    # res(f"{st_name}.sc", '9bulk_eos', list(range(1,11)) + [100], show = 'fit', analys_type = 'fit_a', up="up2", cluster = 'razor128')
    en = db[f"{st_name}.sc", '9bulk_eos', 100].e0_at
    
    print(f"{st_name}: {en:1.3f}")
    


# In[ ]:





# In[34]:


write_database()


# In[ ]:





# In[ ]:





# In[ ]:





# ## Write DFT energies in CSV

# In[35]:


layered_groups = ["P1", "P2_1", "P-1", "Pc", "P2_1_c"]

names_list = ["layered_p1", "layered_p-1", "layered_p21", "layered_p2c",
    "layered_p21c", "alpha", "beta", "gamma", "delta"]


# In[36]:


energies = []

for group in layered_groups:
    if (group == "P2_1_c"):
        calc = db[f"layered_{group}_final_su", '9bulk', 1]
    else:
        calc = db[f"layered_{group}_final", '9bulk', 1]
        
    energies.append(calc.e0)

for st_name in ["alpha", "beta", "gamma", "delta"]:
    en = db[f"{st_name}.sc", '9bulk_eos', 100].e0
    energies.append(en)
    


# In[ ]:





# In[ ]:





# In[37]:


energies


# In[ ]:





# In[38]:


path_energies_dft = '/home/a.burov/icys_2025/niohf/optimized/energies_dft.csv'


# In[39]:


# choose keys and order
fieldnames = ['name', 'energy_dft']

rows = []


with open(path_energies_dft, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(fieldnames)
    for idx in range(len(energies)):
        w.writerow([names_list[idx], energies[idx], ])
        

    


# In[ ]:





# ## Calculated energies

# In[40]:


import pandas as pd


# In[245]:


path_data = "/home/a.burov/icys_2025/niohf/optimized/"


# In[246]:


data_all = {
    "dft": [],
    "mace": [],
    "grace": [],
    "mattersim": [],
    "sevennet": [],
    "chgnet": [],
    "nequip": [],
    "alignn": [],
    "m3gnet": [],
    "upet": [],
    "fairchem": [],
    "tace": [],
    "prophet": [],
}


# In[247]:


missing = []
for type_calc in list(data_all.keys()):
    data_path = f"{path_data}/energies_{type_calc}.csv"
    if not os.path.isfile(data_path):
        print(f"Skipping {type_calc}: no {data_path}")
        missing.append(type_calc)
        continue
    data = pd.read_csv(data_path)
    names = data["name"].to_list()
    values = np.array(data[f"energy_{type_calc}"].to_list()) / 16
    data_all[type_calc] = values

for type_calc in missing:
    del data_all[type_calc]
    
    


# In[248]:


data_all


# In[ ]:





# In[ ]:





# In[249]:


# normalize energies
for type_calc, val in data_all.items():
    data_all[type_calc] = val - min(val)


# In[ ]:





# In[250]:


names


# In[ ]:





# In[251]:


import sys
from pathlib import Path

def _repo_root():
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "plot_colors.py").is_file():
            return cand
    raise FileNotFoundError("plot_colors.py not found from " + str(here))

sys.path.insert(0, str(_repo_root()))
from plot_colors import series_colors

_cols = series_colors(len(data_all))
colors = {name: col for name, col in zip(data_all, _cols)}


# In[252]:


names_plot = [
                r"$P1$-L",
                r"$P\overline{1}$-L", 
                r"$P2_1$-L", 
                r"$P2/c$-L", 
                r"$P2_1/c$-L",  
              r"$\alpha$-F", 
              r"$\beta$-F", 
              r"$\gamma$-F", 
              r"$\delta$-F"
]



# In[262]:


fontsize = 14
lw = 2.0

fig, ax = plt.subplots(1, 1, figsize=(9, 5), dpi=600)
plt.tight_layout()


# Add titles and labels
ax.set_ylabel(r'$E - E_{\mathrm{min}}$, meV/atoms', fontsize=fontsize)
ax.set_xlabel(r'Ni(OH)F phase', fontsize=fontsize)

ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.tick_params(axis='both', which='minor', labelsize=fontsize-2)
ax.yaxis.get_offset_text().set_fontsize(10)
ax.xaxis.set_tick_params(width=2, length=7)
ax.yaxis.set_tick_params(width=2, length=7)

length = len(names)
ax.set_xticks(range(length), names_plot, rotation=45)
idx_list = np.array(range(length))
shift = - (length // 2) + (length % 2)  # initial shift for bars
width = 0.075

for idx, (type_calc, val) in enumerate(data_all.items()):
    data_plot = data_all[type_calc]
    ax.bar(idx_list + width*(shift+idx), data_plot*1e3, width=width, color=colors[type_calc],
                           edgecolor=colors[type_calc], alpha=0.3, label=type_calc)

    # Add crosses for zero energy structures
    zero_mask = np.array(data_plot) == 0
    zero_positions = idx_list[zero_mask] + width*(shift+idx)
    if len(zero_positions) > 0:
        ax.scatter(zero_positions, np.zeros(len(zero_positions))+1.1, marker='x', s=15, 
                   color=colors[type_calc], linewidths=2, zorder=10, alpha=0.3, edgecolor=colors[type_calc])


ax.legend(fontsize=fontsize-4, edgecolor="black", ncols=length//2)

fig.tight_layout()

plt.show()

# fig.savefig("/home/a.burov/icys_2025/niohf/data/figures/energies.png", bbox_inches='tight', dpi=450)
# fig.savefig("/home/a.burov/icys_2025/niohf/data/figures/energies.pdf", bbox_inches='tight', dpi=450)

fig.savefig("/home/a.burov/icys_2025/niohf/data/figures/energies.png", dpi=450)
fig.savefig("/home/a.burov/icys_2025/niohf/data/figures/energies.pdf", dpi=450)




# In[ ]:





# In[ ]:





# ## Map of guesses

# In[263]:


data_all


# In[264]:


list(data_all.keys())


# In[265]:


names_calc = list(data_all.keys())


# In[266]:


names_plot = [
    # "",
    '$P1$-L',
 '$P\\overline{1}$-L',
 '$P2_1$-L',
 '$P2/c$-L',
 '$P2_1/c$-L',
 '$\\alpha$-F',
 '$\\beta$-F',
 '$\\gamma$-F',
 '$\\delta$-F'
]



# In[268]:


# Use CONSISTENT method names
methods = list(data_all.keys())  # 11 methods
n_methods = len(methods)  # Match your data

# ... your heatmap_data creation ...

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(heatmap_data[:, :n_methods], cmap=cmap, aspect='auto', vmin=0, vmax=2)  # Match columns

# Gridlines FIRST
ax.set_xticks(np.arange(n_methods + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(n_structures + 1) - 0.5, minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

# REMOVED MaxNLocator - use exact ticks
ax.set_xticks(np.arange(n_methods))  # Exactly 11 ticks
ax.set_xticklabels(methods, fontsize=18, rotation=45, ha='right')  # Exactly 11 labels
ax.set_xticklabels(methods, fontsize=18, )  # Exactly 11 labels

ax.set_yticks(list(range(0,len(names_plot))))  # Exactly 11 ticks
ax.set_yticklabels(names_plot)
ax.axhline(y=4.5, color='red', linewidth=3, zorder=10)
ax.set_title('Most Stable Structure per Optimization Method', fontsize=22)

plt.tight_layout()

fig.savefig("/home/a.burov/icys_2025/niohf/data/figures/guess_map.png", dpi=450)
fig.savefig("/home/a.burov/icys_2025/niohf/data/figures/guess_map.pdf", dpi=450)

plt.show()





# In[ ]:





# In[ ]:





# In[ ]:





# ### Atomic positions

# In[208]:


st_names = ["layered_p1", "layered_p-1", "layered_p2c", "layered_p21", "layered_p21c", "alpha", "beta", "gamma", "delta"]





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


path_dft = "/home/a.burov/icys_2025/niohf/optimized/dft/"


# In[220]:


# Dictionary of ALL potentials
potentials = {
    "sevennet": "/home/a.burov/icys_2025/niohf/optimized/sevenn/",
    "chgnet": "/home/a.burov/icys_2025/niohf/optimized/chgnet/",
    "alignn": "/home/a.burov/icys_2025/niohf/optimized/alignn/",
    "fairchem": "/home/a.burov/icys_2025/niohf/optimized/fairchem/",
    "grace": "/home/a.burov/icys_2025/niohf/optimized/grace/",
    "m3gnet": "/home/a.burov/icys_2025/niohf/optimized/m3gnet/",
    "mace": "/home/a.burov/icys_2025/niohf/optimized/mace/",
    "mattersim": "/home/a.burov/icys_2025/niohf/optimized/mattersim/",
    "nequip": "/home/a.burov/icys_2025/niohf/optimized/nequip/",
    "upet": "/home/a.burov/icys_2025/niohf/optimized/upet/",
    "tace": "/home/a.burov/icys_2025/niohf/optimized/tace/",
    "prophet": "/home/a.burov/icys_2025/niohf/optimized/prophet/",
}


# In[221]:


_available = {}
for pot_name, pot_path in potentials.items():
    missing_files = [
        name for name in st_names
        if not os.path.isfile(os.path.join(pot_path, name + ".vasp"))
    ]
    if missing_files:
        print(f"Skipping {pot_name}: missing {', '.join(missing_files)}")
        continue
    _available[pot_name] = pot_path
potentials = _available

diffs_dir = {}

for name in st_names:
    st_dft = smart_structure_read(path_dft + name + ".vasp")
    coords_dft = np.array(st_dft.xred)
    
    diffs_dir[name] = {}
    
    # Loop over ALL potentials
    for pot_name, pot_path in potentials.items():
        st_pot = smart_structure_read(pot_path + name + ".vasp")
        coords_pot = np.array(st_pot.xred)
        
        # Calculate fractional differences (minimum image)
        frac_diff = coords_pot - coords_dft
        frac_diff_minimg = frac_diff - np.round(frac_diff)
        per_atom_distances = np.sum(frac_diff_minimg**2, axis=1)**0.5
        
        diffs_dir[name][pot_name] = per_atom_distances
    
    print(f"Processed {name}: {list(potentials.keys())}")


# In[ ]:





# In[ ]:





# In[234]:


potentials_list = list(potentials.keys())  # ['sevennet', 'chgnet', 'alignn', ...]
n_pots = len(potentials_list)  # 10

fontsize = 18
lw = 2.0

phases = list(diffs_dir.keys())
data_all = {}  # Collect all data
for pot in potentials_list:
    data_all[pot] = [diffs_dir[p][pot] for p in phases]

x = np.arange(len(phases), dtype=float)
width = 0.85 / n_pots  # Dynamic width for 10 boxes

fig, ax = plt.subplots(figsize=(15, 6))  # Wider figure

# Colors for 10 potentials
colors = plt.cm.tab10(np.linspace(0, 1, n_pots))  # 10 distinct colors
box_plots = []

# Loop over ALL potentials
for i, pot_name in enumerate(potentials_list):
    bp = ax.boxplot(data_all[pot_name], 
                   positions=x + (i - n_pots/2 + 0.5) * width,  # Center groups
                   widths=width*0.8, patch_artist=True, showmeans=True)
    box_plots.append(bp)
    
    # Color styling
    box_color = colors[i]
    for b in bp['boxes']:
        b.set_facecolor(box_color)
        b.set_alpha(0.4)
    for part in ['caps', 'whiskers']:
        [item.set(color=box_color, linewidth=lw) for item in bp[part]]
    for median in bp['medians']:
        median.set(color="black", linewidth=lw)
    for mean in bp['means']:
        mean.set(marker='s', markeredgecolor="black", markerfacecolor="black", markersize=3)


# Formatting
ax.set_xticks(x)
ax.set_xticklabels(names_plot, rotation=45, ha='right')
ax.tick_params(axis='both', which='major', labelsize=fontsize)
ax.set_xlabel('Ni(OH)F phase', fontsize=fontsize)
ax.set_ylabel('Atomic displacement (fractional)', fontsize=fontsize)
ax.yaxis.get_offset_text().set_fontsize(10)

# Legend using first box of each
legend_elements = [bp['boxes'][0] for bp in box_plots]
ax.legend(legend_elements, potentials_list, loc=1, fontsize=fontsize-2, edgecolor='black', ncols=len(potentials.keys())//4)

plt.tight_layout()
plt.savefig("/home/a.burov/icys_2025/niohf/data/figures/atomic_pos_all.png", dpi=450, bbox_inches='tight')
plt.savefig("/home/a.burov/icys_2025/niohf/data/figures/atomic_pos_all.pdf", dpi=450, bbox_inches='tight')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# ### RMSE errors 

# In[243]:


potentials_list = list(potentials.keys())  # ['sevennet', 'chgnet', 'alignn', ...]
phases = list(diffs_dir.keys())
n_pots = len(potentials_list)
n_struct_types = 2  # layered, dispersed

# Split structures into layered (0-4) and dispersed (5-8)
layered_phases = phases[:5]
dispersed_phases = phases[5:]

# Calculate RMSE for each potential + structure type
rmse_data = {}
for pot in potentials_list:
    # RMSE = sqrt(mean(per_atom_distance**2))
    layered_rmse = [np.sqrt(np.mean(diffs_dir[p][pot]**2)) for p in layered_phases]
    dispersed_rmse = [np.sqrt(np.mean(diffs_dir[p][pot]**2)) for p in dispersed_phases]
    rmse_data[pot] = {'layered': np.mean(layered_rmse), 'dispersed': np.mean(dispersed_rmse)}

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(n_pots)
width = 0.35
fontsize = 18

# Layered bars (left)
layered_vals = [rmse_data[pot]['layered'] for pot in potentials_list]
c_layered, c_dispersed = series_colors(2)
bars1 = ax.bar(x - width/2, layered_vals, width, label='Layered', alpha=0.4, edgecolor=c_layered, color=c_layered)

# Dispersed bars (right)  
dispersed_vals = [rmse_data[pot]['dispersed'] for pot in potentials_list]
bars2 = ax.bar(x + width/2, dispersed_vals, width, label='Dispersed', alpha=0.4, edgecolor=c_dispersed, color=c_dispersed)

# Formatting
ax.set_xlabel('UMLIP type', fontsize=fontsize)
ax.set_ylabel('RMSE (fractional coordinates)', fontsize=fontsize)
ax.set_title('Atomic Position RMSE: Layered vs Dispersed Structures', fontsize=fontsize+2)
ax.set_xticks(x)
ax.set_xticklabels(potentials_list, rotation=45, ha='right')
ax.tick_params(axis='both', labelsize=fontsize)
ax.legend(fontsize=fontsize, loc=7)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
# for bar in bars1:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
#             f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
# for bar in bars2:
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
#             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("/home/a.burov/icys_2025/niohf/data/figures/rmse_layered_dispersed.png", dpi=450, bbox_inches='tight')
plt.savefig("/home/a.burov/icys_2025/niohf/data/figures/rmse_layered_dispersed.pdf", dpi=450, bbox_inches='tight')
plt.show()




# In[ ]:





# In[ ]:





# ## Attempt to increase symmetry

# In[24]:


from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure


# In[34]:


# Load your structure
# structure = Structure.from_file('/home/a.burov/icys_2025/niohf/optimized/sevenn/layered.cif')
structure = Structure.from_file('/home/a.burov/icys_2025/niohf/optimized/dft/delta.cif')

# structure = Structure.from_file('/home/a.burov/icys_2025/niohf/initial_no_relaxation/CuOHF.cif')



# In[35]:


# Symmetrize with looser tolerance
sga = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=30)
symmetrized_structure = sga.get_refined_structure()

# Check symmetry improvement
print(f"Original space group: {SpacegroupAnalyzer(structure, symprec=0.01).get_space_group_symbol()}")
print(f"Refined space group: {sga.get_space_group_symbol()}")

# Save
# symmetrized_structure.to(filename='symmetrized.cif')


# In[ ]:





# In[29]:


symmetrized_structure


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





write_database()


# In[ ]:


# MP setup


# In[ ]:


st = st_init.copy()


# In[ ]:


add("layered", '9_bulk_mp', 1, it_folder = 'inter', input_st = st, cluster = 'razor128', up='up2', run=2)


# In[ ]:





# In[ ]:





# ## Defects

# In[ ]:


get_ipython().system('pip install pymatgen-analysis-defects')



# In[ ]:





# In[ ]:


from pymatgen.io.vasp import Chgcar


# In[ ]:


from pymatgen.analysis.defects.generators import ChargeInterstitialGenerator


# In[ ]:


cig = ChargeInterstitialGenerator()


# In[ ]:


chg_path = "/home/a.burov/icys_2025/niohf/layered/layered.sc.9bulk_eos/100.CHGCAR"


# In[ ]:


chg = Chgcar.from_file(chg_path)
    


# In[ ]:


defects = list(cig.generate(chg, insert_species=["Li"]))


# In[ ]:





# In[ ]:




