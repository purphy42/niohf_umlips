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





# In[ ]:





# In[6]:


from matplotlib import font_manager as fm
font_path = "/home/a.burov/fonts/ARIAL.TTF"
fm.fontManager.addfont(font_path)


# In[7]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[8]:


np.set_printoptions(precision=5)


# In[9]:


# %matplotlib inline
plt.rcParams['figure.dpi'] = 450


# In[10]:


API_KEY = "LTSM6dStBrl69FjopxP7KdZBP35B1yh7"


# In[ ]:





# In[ ]:





# In[ ]:





# ## Process and visualize phonon calculations

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import subprocess
import os
import yaml


# In[ ]:


from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import subprocess
import matplotlib.pyplot as plt
import os


# In[13]:


base_path_all = [
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/layered_P1"),
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/layered_P-1/"),
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/layered_Pc"),
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/layered_P2_1/"),
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/layered_P2_1_c/"),
    
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/diaspore_alpha/"),
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/diaspore_beta/"),
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/diaspore_gamma/"),
    Path("/home/a.burov/icys_2025/niohf/phonopy_calc/diaspore_delta/"),
]


# In[14]:





# In[15]:


def process_phonopy_volume(vol_dir, save_csv=True):
    """
    Process a single volume directory for phonopy calculations.
    
    Parameters:
    -----------
    vol_dir : Path
        Path to volume directory (e.g., vol_100) containing 0,1,2,... subdirectories
    save_csv : bool
        Whether to save results to CSV file
    
    Returns:
    --------
    dict or None
        Dictionary containing temperatures, gibbs energies, volume, etc.
    """
    vol_name = vol_dir.name
    print(f"\n  🔬 Processing {vol_name}")
    original_cwd = os.getcwd()
    
    try:
        os.chdir(vol_dir)
        
        # 1. Find all numbered directories (0, 1, 2, ... N)
        all_dirs = sorted([d for d in vol_dir.glob("*") if d.is_dir() and d.name.isdigit()], 
                          key=lambda x: int(x.name))
        
        if not all_dirs:
            print(f"    ❌ No numbered directories found in {vol_name}")
            return None
        
        # Separate zero directory and displacement directories
        zero_dir = vol_dir / "0"
        disp_dirs = [d for d in all_dirs if d != zero_dir]
        
        # Check for vasprun.xml files
        all_vaspruns = [d / "vasprun.xml" for d in disp_dirs if (d / "vasprun.xml").exists()]
        
        print(f"    📁 Found {len(all_vaspruns)} vaspruns (displacements 1-{len(disp_dirs)})")
        
        if len(all_vaspruns) < 12:
            print(f"    ❌ Need ≥12 displacements, found {len(all_vaspruns)}")
            return None
        
        # 2. Test each vasprun in order
        good_vaspruns = []
        for i, vasprun in enumerate(all_vaspruns):
            dir_num = disp_dirs[i].name
            result = subprocess.run(["phonopy", "-f", str(vasprun)], 
                                  capture_output=True, cwd=vol_dir, timeout=15)
            if result.returncode == 0:
                good_vaspruns.append(vasprun)
                print(f"      ✓ Displacement {dir_num}")
            else:
                print(f"      ✗ Displacement {dir_num} (skipped)")
        
        print(f"    📊 Good displacements: {len(good_vaspruns)}/{len(all_vaspruns)}")
        
        if len(good_vaspruns) < 12:
            print(f"    ❌ Need ≥12 good displacements")
            return None
        
        # 3. Create FORCE_SETS with good displacements in order
        print(f"    🔄 Running phonopy -f on {len(good_vaspruns)} files...")
        subprocess.run(["phonopy", "-f"] + [str(v) for v in good_vaspruns], 
                       cwd=vol_dir, capture_output=True)
        
        # 4. Create phonopy configuration
        conf = """DIM = 1 1 1
            DISPLACEMENT_DISTANCE = 0.01
            MP = 16 16 16
            TSTEP = 10
            TMAX = 1000
        """
        with open("phonopy.conf", "w") as f:
            f.write(conf)
        
        # 5. Generate FORCE_CONSTANTS
        print(f"    🔄 Generating FORCE_CONSTANTS...")
        subprocess.run(["phonopy", "phonopy.conf"], cwd=vol_dir, capture_output=True)
        
        # 6. Calculate thermal properties
        print(f"    🔄 Calculating thermal properties...")
        subprocess.run(["phonopy", "phonopy.conf", "-t", "-p"], cwd=vol_dir, capture_output=True)
        
        # 7. Read results
        tp_file = vol_dir / "thermal_properties.yaml"
        if not tp_file.exists():
            print(f"    ❌ thermal_properties.yaml not found")
            return None
        
        with open(tp_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract volume number from directory name
        volume = float(vol_name.split('_')[-1])
        
        result = {
            'temperatures': np.array(data['temperature']),
            'gibbs_kJmol': np.array(data['gibbs_free_energy']),
            'entropy_J_mol_K': np.array(data.get('entropy', [])),
            'heat_capacity_J_mol_K': np.array(data.get('heat_capacity', [])),
            'volume': volume,
            'good_disps': len(good_vaspruns),
            'volume_name': vol_name,
            'phase_name': vol_dir.parent.name
        }
        
        # 8. Save to CSV
        if save_csv:
            df = pd.DataFrame({
                'temperature_K': result['temperatures'],
                'gibbs_free_energy_kJ_mol': result['gibbs_kJmol'],
                'entropy_J_mol_K': result['entropy_J_mol_K'],
                'heat_capacity_J_mol_K': result['heat_capacity_J_mol_K'],
                'volume_A3': result['volume'],
                'good_displacements': result['good_disps']
            })
            csv_path = vol_dir / f"{vol_name}_thermal_properties.csv"
            df.to_csv(csv_path, index=False)
            print(f"    💾 Saved CSV: {csv_path.name}")
            result['csv_path'] = csv_path
        
        print(f"    ✅ Complete: {len(result['temperatures'])} temperature points")
        return result
        
    except Exception as e:
        print(f"    ❌ Error processing {vol_name}: {str(e)}")
        return None
    
    finally:
        os.chdir(original_cwd)


def process_phase(phase_path, save_csv=True, max_volumes=None):
    """
    Process all volumes for a single phase.
    
    Parameters:
    -----------
    phase_path : Path
        Path to phase directory (e.g., layered_P1)
    save_csv : bool
        Save individual CSVs
    max_volumes : int, optional
        Maximum number of volumes to process
    
    Returns:
    --------
    dict
        Dictionary with volume names as keys and results as values
    """
    phase_name = phase_path.name
    print(f"\n{'='*60}")
    print(f"📂 Processing phase: {phase_name}")
    print(f"{'='*60}")
    
    # Find all volume directories (vol_*)
    vol_dirs = sorted([d for d in phase_path.iterdir() if d.is_dir() and d.name.startswith('vol_')],
                     key=lambda x: int(x.name.split('_')[-1]))
    
    if not vol_dirs:
        print(f"  ⚠️  No volume directories found in {phase_name}")
        return {}
    
    print(f"  Found {len(vol_dirs)} volumes: {[d.name for d in vol_dirs]}")
    
    if max_volumes:
        vol_dirs = vol_dirs[:max_volumes]
        print(f"  Processing first {max_volumes} volumes")
    
    # Process each volume
    results = {}
    for vol_dir in vol_dirs:
        result = process_phonopy_volume(vol_dir, save_csv=save_csv)
        if result:
            results[vol_dir.name] = result
    
    print(f"\n  ✅ {phase_name}: Successfully processed {len(results)}/{len(vol_dirs)} volumes")
    return results


def process_all_phases(base_path, phases=None, save_individual_csv=True, save_combined_csv=True, max_volumes_per_phase=None):
    """
    Process all phases or specified phases.
    
    Parameters:
    -----------
    base_path : Path
        Base directory containing phase folders
    phases : list, optional
        List of phase names to process (e.g., ['layered_P1', 'diaspore_alpha'])
        If None, process all directories
    save_individual_csv : bool
        Save CSV for each volume
    save_combined_csv : bool
        Save combined CSV for each phase
    max_volumes_per_phase : int, optional
        Maximum number of volumes to process per phase
    
    Returns:
    --------
    dict
        Dictionary with phase names as keys and volume results as values
    """
    base_path = Path(base_path)
    all_results = {}
    
    # Get all phase directories
    if phases:
        phase_dirs = [base_path / p for p in phases if (base_path / p).is_dir()]
    else:
        # Get all subdirectories
        phase_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"\n{'#'*60}")
    print(f"STARTING PHONOPY PROCESSING")
    print(f"Base path: {base_path}")
    print(f"Found {len(phase_dirs)} phases: {[d.name for d in phase_dirs]}")
    print(f"{'#'*60}")
    
    # Process each phase
    for phase_dir in phase_dirs:
        phase_results = process_phase(phase_dir, save_csv=save_individual_csv, max_volumes=max_volumes_per_phase)
        if phase_results:
            all_results[phase_dir.name] = phase_results
            
            # Save combined CSV for this phase
            if save_combined_csv and phase_results:
                all_data = []
                for vol_name, data in phase_results.items():
                    df_temp = pd.DataFrame({
                        'volume_name': vol_name,
                        'volume_A3': data['volume'],
                        'temperature_K': data['temperatures'],
                        'gibbs_free_energy_kJ_mol': data['gibbs_kJmol'],
                        'entropy_J_mol_K': data['entropy_J_mol_K'],
                        'heat_capacity_J_mol_K': data['heat_capacity_J_mol_K'],
                        'good_displacements': data['good_disps']
                    })
                    all_data.append(df_temp)
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    combined_csv = phase_dir / f"{phase_dir.name}_all_volumes.csv"
                    combined_df.to_csv(combined_csv, index=False)
                    print(f"\n  💾 Saved combined CSV for {phase_dir.name}: {combined_csv}")
    
    return all_results


def plot_gibbs_comparison(all_results, base_path, save_plots=True):
    """
    Plot Gibbs free energies comparing different phases and volumes.
    """
    if not all_results:
        print("No data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    # Generate distinct colors for phases
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_results)))
    
    # Store for analysis
    phase_min_gibbs = {}
    phase_temps = {}
    
    # Plot 1: Minimum Gibbs per phase (best volume at each T)
    for idx, (phase_name, phase_data) in enumerate(all_results.items()):
        if not phase_data:
            continue
        
        color = colors[idx]
        
        # Get temperatures (assuming all volumes have same T points)
        temps = list(phase_data.values())[0]['temperatures']
        phase_temps[phase_name] = temps
        
        # Find minimum Gibbs across volumes at each temperature
        min_gibbs_phase = np.min([data['gibbs_kJmol'] for data in phase_data.values()], axis=0)
        phase_min_gibbs[phase_name] = min_gibbs_phase
        
        ax1.plot(temps, min_gibbs_phase, '-', lw=2.5, color=color, label=phase_name)
        
        # Plot 2: All volumes for this phase
        for vol_name, data in phase_data.items():
            ax2.plot(temps, data['gibbs_kJmol'], '-', lw=1, color=color, alpha=0.3)
        # Overlay minimum line
        ax2.plot(temps, min_gibbs_phase, '-', lw=2.5, color=color, label=phase_name)
    
    ax1.set_xlabel('Temperature (K)', fontsize=12)
    ax1.set_ylabel('Gibbs Free Energy (kJ/mol)', fontsize=12)
    ax1.set_title('Minimum Gibbs Energy per Phase', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Temperature (K)', fontsize=12)
    ax2.set_ylabel('Gibbs Free Energy (kJ/mol)', fontsize=12)
    ax2.set_title('All Volumes (thin) with Minimum (thick)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10, ncol=2)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Relative to global minimum
    if phase_min_gibbs:
        # Find global minimum across all phases at each T
        all_min = np.min(list(phase_min_gibbs.values()), axis=0)
        
        for phase_name, min_gibbs in phase_min_gibbs.items():
            rel_gibbs = min_gibbs - all_min
            ax3.plot(phase_temps[phase_name], rel_gibbs, '-', lw=2.5, 
                    color=colors[list(all_results.keys()).index(phase_name)], 
                    label=phase_name)
        
        ax3.set_xlabel('Temperature (K)', fontsize=12)
        ax3.set_ylabel('Relative Gibbs Energy (kJ/mol)', fontsize=12)
        ax3.set_title('Relative to Global Minimum', fontsize=14, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(alpha=0.3)
        
        # Plot 4: Phase stability diagram
        phase_names = list(phase_min_gibbs.keys())
        stability = np.argmin(list(phase_min_gibbs.values()), axis=0)
        
        for idx, phase_name in enumerate(phase_names):
            mask = (stability == idx)
            if np.any(mask):
                temps_stable = phase_temps[phase_name][mask]
                ax4.plot(temps_stable, [idx] * len(temps_stable), 'o-', 
                        color=colors[idx], markersize=8, linewidth=2, label=phase_name)
        
        ax4.set_xlabel('Temperature (K)', fontsize=12)
        ax4.set_ylabel('Stable Phase', fontsize=12)
        ax4.set_yticks(range(len(phase_names)))
        ax4.set_yticklabels(phase_names, fontsize=10)
        ax4.set_title('Phase Stability Diagram', fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_plots:
        png_path = base_path / 'phase_stability_analysis.png'
        pdf_path = base_path / 'phase_stability_analysis.pdf'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"\n📊 Saved plots: {png_path}, {pdf_path}")
    
    plt.show()
    return fig



# In[ ]:





# In[16]:


# Your base path
base_path = Path("/home/a.burov/icys_2025/niohf/phonopy_calc")

# Define all phases (based on your directory listing)
all_phases = [
    "layered_P1",
    "layered_P-1", 
    "layered_P2_1",
    "layered_P2_1_c",
    "layered_Pc",
    "diaspore_alpha",
    "diaspore_beta",
    "diaspore_gamma",
    "diaspore_delta"
]

# Process all phases
print("Starting phonopy processing for all phases...")
print(f"Phases to process: {all_phases}")

results = process_all_phases(
    base_path,
    phases=all_phases,
    save_individual_csv=True,
    save_combined_csv=True,
    max_volumes_per_phase=None  # Process all volumes (or set to e.g., 5 for testing)
)

# Plot and analyze results
if results:
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - GENERATING PLOTS")
    print("="*80)
    
    plot_gibbs_comparison(results, base_path, save_plots=True)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for phase_name, phase_data in results.items():
        print(f"\n📁 {phase_name}:")
        print(f"   Volumes processed: {len(phase_data)}")
        
        # Find best volume (lowest Gibbs at 0K)
        best_vol = min(phase_data.items(), 
                      key=lambda x: x[1]['gibbs_kJmol'][0])
        print(f"   Best volume at 0K: {best_vol[0]} ({best_vol[1]['volume']:.1f} Å³)")
        print(f"     G(0K) = {best_vol[1]['gibbs_kJmol'][0]:.2f} kJ/mol")
        print(f"     G(300K) = {best_vol[1]['gibbs_kJmol'][30]:.2f} kJ/mol")
        print(f"     G(500K) = {best_vol[1]['gibbs_kJmol'][50]:.2f} kJ/mol")
    
    # Find overall best phase at different temperatures
    print("\n" + "="*80)
    print("PHASE STABILITY SUMMARY")
    print("="*80)
    
    # Get temperatures from first phase
    first_phase = list(results.values())[0]
    first_vol = list(first_phase.values())[0]
    temps = first_vol['temperatures']
    
    for temp in [0, 300, 500, 800, 1000]:
        temp_idx = np.argmin(np.abs(temps - temp))
        best_phase = min(results.keys(), 
                       key=lambda p: np.min([data['gibbs_kJmol'][temp_idx] 
                                            for data in results[p].values()]))
        best_energy = min([np.min([data['gibbs_kJmol'][temp_idx] 
                                  for data in results[p].values()]) 
                          for p in results.keys()])
        print(f"\nAt {temp} K:")
        print(f"  Most stable phase: {best_phase}")
        print(f"  Gibbs energy: {best_energy:.2f} kJ/mol")
else:
    print("\n❌ No results were generated. Please check your directory structure.")

    


# In[ ]:





# In[ ]:


print(gibbs_data)


# In[ ]:





# In[ ]:





# In[ ]:




