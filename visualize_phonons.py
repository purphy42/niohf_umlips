#!/usr/bin/env python
# coding: utf-8

# In[1]:


from siman.geo import calc_kspacings


# In[2]:


from ase.optimize import FIRE  
from pathlib import Path
import csv
import shutil 
import subprocess
import glob


# ## Other default imports 

# In[3]:


import os, re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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



# In[4]:


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



# In[5]:


from ase.optimize import FIRE  
from ase.io import read, write
from ase.visualize import view


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





# ## Process and visualize phonon calculations

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import subprocess
import os


# In[11]:


base_path = Path("/home/a.burov/icys_2025/niohf/phonopy_calc/layered_P1")
volume_dirs = sorted(base_path.glob("vol_*"))


# In[1]:


def process_phonopy_volume(vol_dir):
    vol_name = vol_dir.name
    print(f"\n🔬 {vol_name}")
    os.chdir(vol_dir)
    
    # 1. EXCLUDE DIR=0, get displacements 1-N in CORRECT NUMERICAL ORDER
    all_dirs = sorted([d for d in vol_dir.glob("[0-9]*") if d.is_dir()], 
                      key=lambda x: int(x.name))
    zero_dir = vol_dir / "0"
    
    # STRICT ORDER: dir 1 → disp #1, dir 2 → disp #2, etc.
    disp_dirs = [d for d in all_dirs if d != zero_dir]
    all_vaspruns = [d / "vasprun.xml" for d in disp_dirs if (d / "vasprun.xml").exists()]
    
    print(f"  📁 Found {len(all_vaspruns)} potential vaspruns (dirs 1-N)")
    
    # 2. ✅ TEST IN ORDER - keep only good files maintaining sequence
    good_vaspruns = []
    for i, vasprun in enumerate(all_vaspruns):
        dir_num = disp_dirs[i].name
        result = subprocess.run(["phonopy", "-f", str(vasprun)], 
                              capture_output=True, cwd=vol_dir, timeout=15)
        if result.returncode == 0:
            good_vaspruns.append(vasprun)
            print(f"    ✓ {dir_num}")
        else:
            print(f"    ✗ {dir_num} (position mismatch - skipped)")
    
    print(f"  📊 Good: {len(good_vaspruns)}/{len(all_vaspruns)} in correct order")
    
    if len(good_vaspruns) < 12:
        print(f"  ❌ Need ≥12 consecutive good displacements")
        return None
    
    # 3. CRITICAL: Process in EXACT SAME ORDER
    print("  🔄 phonopy -f (ordered good files)...")
    subprocess.run(["phonopy", "-f"] + [str(v) for v in good_vaspruns], 
                   cwd=vol_dir, capture_output=True)
    print("  ✓ FORCE_SETS created (order preserved)")
    
    # 4. Primitive cell config
    conf = """
DIM = 1 1 1
DISPLACEMENT_DISTANCE = 0.01
MP = 16 16 16
TSTEP = 10
TMAX = 1000
"""
    with open("phonopy.conf", "w") as f:
        f.write(conf)
    
    # 5. Generate FORCE_CONSTANTS
    print("  🔄 phonopy phonopy.conf...")
    subprocess.run(["phonopy", "phonopy.conf"], cwd=vol_dir)
    print("  ✓ FORCE_CONSTANTS")
    
    # 6. Gibbs energy
    print("  🔄 Thermal properties...")
    subprocess.run(["phonopy", "phonopy.conf", "-t", "-p"], cwd=vol_dir)
    
    tp_file = vol_dir / "thermal_properties.yaml"
    if tp_file.exists():
        import yaml
        with open(tp_file) as f:
            data = yaml.safe_load(f)
        print(f"  ✅ G(T): {len(data['temperature'])} points ({len(good_vaspruns)} disps)")
        return {
            'temperatures': np.array(data['temperature']),
            'gibbs_kJmol': np.array(data['gibbs_free_energy']),
            'volume': float(vol_name.split('_')[1]),
            'good_disps': len(good_vaspruns)
        }
    return None



# In[ ]:





# In[ ]:


# 🔥 EXECUTE
base_path = Path("/home/a.burov/icys_2025/niohf/phonopy_calc/layered_P1")
gibbs_data = {}

for vol_dir in sorted(base_path.glob("vol_*"))[:5]:
    result = process_phonopy_volume(vol_dir)
    if result:
        gibbs_data[vol_dir.name] = result

# Plot
if gibbs_data:
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = series_colors(len(gibbs_data))
    
    for i, (vol_name, data) in enumerate(gibbs_data.items()):
        ax.plot(data['temperatures'], data['gibbs_kJmol'], 
                '-', lw=3, color=colors[i], 
                label=f'vol_{int(data["volume"])} ({data["good_disps"]} disps)')
    
    temps = list(gibbs_data.values())[0]['temperatures']
    min_gibbs = np.array([min(d['gibbs_kJmol'][i] for d in gibbs_data.values()) 
                         for i in range(len(temps))])
    ax.plot(temps, min_gibbs, "-", color="#1A1A1A", lw=5, label="Stable phase")
    
    ax.set_xlabel('T (K)'); ax.set_ylabel('G (kJ/mol)')
    ax.set_title('NiOHF QHA - Ordered Displacement Processing', fontsize=16)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(base_path / 'niohf_gibbs_ordered.png', dpi=450)
    plt.show()
    
    print(f"\n✅ {len(gibbs_data)} volumes complete!")



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




