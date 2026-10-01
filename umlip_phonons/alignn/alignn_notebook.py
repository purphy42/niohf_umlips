#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Internal phonopy methods

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.optimize import FIRE, LBFGS
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import yaml
import subprocess

from ase.calculators.eam import EAM

import os
import shutil
import glob
from pathlib import Path

from ase.build import bulk
from ase.constraints import ExpCellFilter

import pandas as pd


# In[2]:


from alignn.ff.ff import AlignnAtomwiseCalculator,default_path
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.atoms import ase_to_atoms


# In[3]:


from phonopy import Phonopy
from phonopy.interface.calculator import get_displacements_and_forces
from phonopy.structure.atoms import PhonopyAtoms


# In[ ]:





# In[4]:


def general_relaxer(ase_atoms="", calculator="", fmax=1e-4, steps=0, relax=True, max_step=0.1):
    ase_atoms.calc = calculator
    if not relax:
         return ase_atoms.get_potential_energy()
        
    atoms_constrained = ExpCellFilter(ase_atoms)
    dyn = FIRE(atoms_constrained, maxstep=max_step, dt=0.1)
    dyn.run(fmax=fmax, steps=steps)
    
    return ase_atoms, dyn



# In[ ]:





# In[5]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[6]:


np.set_printoptions(precision=4)


# In[7]:


# %matplotlib inline
plt.rcParams['figure.dpi'] = 450


# In[ ]:





# In[2]:


path_base = (
    "/home/arseniy/Desktop/work/niohf/optimized/alignn"
    if Path("/home/arseniy/Desktop/work/niohf/optimized/alignn").exists()
    else "/home/a.burov/icys_2025/niohf/optimized/alignn/"
)
_UMLP_OUT = (
    Path("/home/arseniy/Desktop/work/niohf/umlip_phonons")
    if Path("/home/arseniy/Desktop/work/niohf/umlip_phonons").exists()
    else Path("/home/a.burov/icys_2025/niohf/umlip_phonons")
)


# In[9]:


model_path = "/home/a.burov/umlip/alignn/v12.2.2024_mp_1.5mill/"


# In[10]:


alignn_cal = AlignnAtomwiseCalculator(path=model_path, model_filename="best_model.pt")



# In[ ]:





# In[6]:


fontsize = 20


# In[7]:


env_used = "alignn-env"


# In[8]:


potential = "alignn"


# In[ ]:





# In[12]:


def _phonon_fig_dir():
    for p in (
        Path("/home/arseniy/Desktop/work/niohf/figures/phonons"),
        Path("/home/a.burov/icys_2025/niohf/figures/phonons"),
        Path("/home/a.burov/icys_2025/niohf/data/figures/phonons"),
    ):
        if p.exists() or p.parent.exists():
            p.mkdir(parents=True, exist_ok=True)
            return str(p)
    p = Path("/home/arseniy/Desktop/work/niohf/figures/phonons")
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _select_eos_basin(volumes, energies, yaml_files, reverse_tol=0.05, spike_tol=0.20, min_keep=6):
    """
    Keep one convex E(V) bowl around the energy minimum for phonopy-qha.

    ALIGNN often reconstructs at large expansion: E drops again after the
    barrier, and a single point can spike by ~0.5 eV. Both break the QHA fit
    (Gibbs jumps / vertical free-energy minima). Points past a reverse slope
    are a second basin and are dropped. Remaining spikes are removed by
    leave-one-out residuals, but only while the minimum stays interior.
    Fewer than min_keep points (was 4 for gamma) produces discontinuous G(T).
    """
    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)
    yaml_files = list(yaml_files)
    order = np.argsort(v)
    v, e = v[order], e[order]
    yaml_files = [yaml_files[i] for i in order]

    imin = int(np.argmin(e))
    left = imin
    for i in range(imin, 0, -1):
        if e[i - 1] < e[i] - reverse_tol:
            break
        left = i - 1
    right = imin
    for i in range(imin, len(e) - 1):
        if e[i + 1] < e[i] - reverse_tol:
            break
        right = i + 1

    keep = np.zeros(len(e), dtype=bool)
    keep[left:right + 1] = True
    dropped = [
        f"basin V={v[i]:.2f} E={e[i]:.4f}" for i in range(len(e)) if not keep[i]
    ]

    # Drop interior spikes, but never the current minimum and never so many
    # that the minimum is pushed onto the volume edge.
    while int(keep.sum()) > min_keep:
        idx = np.where(keep)[0]
        if len(idx) < 5:
            break
        loo = np.full(len(idx), np.nan)
        for j, i in enumerate(idx):
            if i == int(idx[np.argmin(e[idx])]):
                continue
            m = np.ones(len(idx), dtype=bool)
            m[j] = False
            coef = np.polyfit(v[idx][m], e[idx][m], 2)
            loo[j] = e[i] - np.polyval(coef, v[i])
        if not np.isfinite(loo).any() or np.nanmax(np.abs(loo)) < spike_tol:
            break
        jworst = int(np.nanargmax(np.abs(loo)))
        trial = keep.copy()
        trial[int(idx[jworst])] = False
        tidx = np.where(trial)[0]
        tmin = int(tidx[np.argmin(e[tidx])])
        if tmin in (tidx[0], tidx[-1]):
            break
        keep = trial
        i = int(idx[jworst])
        dropped.append(f"spike V={v[i]:.2f} E={e[i]:.4f} loo={loo[jworst]:+.4f}")

    if dropped:
        print("EOS points excluded: " + ", ".join(dropped))

    v2, e2 = v[keep], e[keep]
    y2 = [yaml_files[i] for i, k in enumerate(keep) if k]
    imin2 = int(np.argmin(e2))
    if len(e2) < min_keep:
        raise RuntimeError(
            f"EOS basin has only {len(e2)} points (need ≥{min_keep}). "
            f"V=[{v2[0]:.2f},{v2[-1]:.2f}]. Narrow the volume window to one "
            f"bowl (avoid reconstructive expansion) and rerun."
        )
    if imin2 in (0, len(e2) - 1):
        side = "compressed" if imin2 == 0 else "expanded"
        raise RuntimeError(
            f"EOS minimum is at the {side} edge "
            f"(V={v2[imin2]:.2f} Å³, n={len(e2)}). "
            f"Widen scale on that side and rerun; QHA on an edge minimum "
            f"produces the discontinuous Gibbs curves."
        )
    return v2, e2, y2


def _relax_ions_fixed_cell(atoms, tag, fmax=1e-4, steps=2000, maxstep=0.05):
    """Relax ions with the cell held fixed.

    FIRE (dt=0.03, 1000 steps) did not converge gamma: several volumes
    reached fmax ~ 1e-3 and then walked uphill, so E(V) was jagged and
    phonopy-qha put a ~2.8 eV jump in G(T) near 190 K. LBFGS with a short
    maxstep stays in that basin and is required to hit fmax.
    """
    atoms.calc = alignn_cal
    opt = LBFGS(
        atoms,
        trajectory=f"{tag}_relax.traj",
        logfile=f"{tag}_opt.log",
        maxstep=maxstep,
    )
    converged = opt.run(fmax=fmax, steps=steps)
    fmax_now = float(np.max(np.linalg.norm(atoms.get_forces(), axis=1)))
    if not converged or fmax_now > fmax:
        raise RuntimeError(
            f"{tag} did not reach fmax={fmax:.1e} "
            f"(fmax={fmax_now:.4e} eV/Å after {steps} LBFGS steps). "
            "An unconverged volume makes the gamma E(V) jagged and G(T) jumps."
        )
    return atoms


def _fixed_volume_energy(equilibrium_atoms, scale_factor, tag, fmax=1e-4, steps=2000):
    """Relax ions at a fixed volume scale and return (atoms, volume, energy)."""
    scaled = equilibrium_atoms.copy()
    scaled.set_cell(equilibrium_atoms.get_cell() * scale_factor ** (1 / 3), scale_atoms=True)
    _relax_ions_fixed_cell(scaled, tag, fmax=fmax, steps=steps)
    return scaled, float(scaled.get_volume()), float(scaled.get_potential_energy())


def _bracket_volume_window(equilibrium_atoms, scale, n_probe=7, max_extend=4, min_width=0.12):
    """
    Cheap E(V) scan, then a phonon window with the minimum strictly inside.

    ALIGNN layered cells keep their lowest energy at the compressed edge of
    [0.76, 1.06]. Extending that edge here avoids 11 full phonon supercells
    that _select_eos_basin would then reject.

    Probe fmax matches the phonon loop (1e-4). A looser or unfinished probe
    for gamma centered the window on an expanded local well; basin-cut then
    left only 4 compressed points and a discontinuous G(T). Unconverged
    FIRE (1000 steps) did the same: volumes walked uphill and G(T) jumped
    by ~2.8 eV near 190 K.
    """
    lo, hi = float(scale[0]), float(scale[1])
    print(f"Probing E(V) before phonons, start {lo:.3f}→{hi:.3f}")
    for attempt in range(max_extend + 1):
        probes = np.linspace(lo, hi, n_probe)
        energies, vols = [], []
        for i, s in enumerate(probes):
            _, vol, en = _fixed_volume_energy(equilibrium_atoms, float(s), f"probe{attempt}_{i}")
            energies.append(en)
            vols.append(vol)
            print(f"  probe {i + 1}/{n_probe} scale={s:.3f} V={vol:.2f} E={en:.4f}")
        imin = int(np.argmin(energies))
        if 0 < imin < n_probe - 1:
            # Keep ≥ min_width around the minimum so QHA has enough points
            # after basin pruning, without swallowing a second reconstructive well.
            half = 0.5 * min_width
            new_lo = max(lo, float(probes[imin]) - half)
            new_hi = min(hi, float(probes[imin]) + half)
            # Prefer probe neighbors when they already span the well.
            neigh_lo = float(probes[max(imin - 1, 0)])
            neigh_hi = float(probes[min(imin + 1, n_probe - 1)])
            new_lo = min(new_lo, neigh_lo)
            new_hi = max(new_hi, neigh_hi)
            if new_hi - new_lo < min_width:
                new_lo = float(probes[max(imin - 2, 0)])
                new_hi = float(probes[min(imin + 2, n_probe - 1)])
            print(
                f"E(V) minimum at scale {probes[imin]:.3f} "
                f"(V={vols[imin]:.2f}); phonon window {new_lo:.3f}→{new_hi:.3f}"
            )
            return new_lo, new_hi
        span = hi - lo
        if imin == 0:
            lo = max(0.50, lo - 0.5 * span)
            print(f"minimum still at the compressed edge; extending probes to {lo:.3f}→{hi:.3f}")
        else:
            hi = min(1.25, hi + 0.5 * span)
            print(f"minimum still at the expanded edge; extending probes to {lo:.3f}→{hi:.3f}")
    raise RuntimeError(
        f"No interior E(V) minimum after probes {lo:.3f}→{hi:.3f}. "
        "ALIGNN energy is still lowest at the edge of this window."
    )


def calc_phonopy(equilibrium_atoms, mul_matrix=[[2,0,0], [0,2,0], [0,0,2]], env_used="msdb", scale=[0.97, 1.10], volume_points=11):

    scale_init, scale_end = _bracket_volume_window(equilibrium_atoms, scale)
    volume_scales = np.linspace(scale_init, scale_end, volume_points)
    volumes = []
    energies = []

    for f in glob.glob("thermal_properties.yaml-*"):
        os.remove(f)
    
    print("Starting QHA calculation with alignn...")
    print(f"Volume scales {scale_init:.3f}→{scale_end:.3f} ({volume_points} pts)")
    
    for i, scale_factor in enumerate(volume_scales):
        print(f"\n{'='*60}")
        print(f"Volume point {i+1}/{volume_points}: scale = {scale_factor:.4f}")
        print(f"{'='*60}")
        
        # Relax ions at fixed volume (no ExpCellFilter → cell stays fixed).
        scaled_atoms, vol, en = _fixed_volume_energy(
            equilibrium_atoms, float(scale_factor), f"vol_{i:02d}"
        )
        volumes.append(vol)
        energies.append(en)
        
        print(f"Volume: {vol:.4f} Å³, Energy: {en:.6f} eV")
        
        # Convert to Phonopy
        unitcell = PhonopyAtoms(
            symbols=scaled_atoms.get_chemical_symbols(),
            cell=scaled_atoms.get_cell(),
            scaled_positions=scaled_atoms.get_scaled_positions()
        )
        
        # Create Phonopy object
        phonon = Phonopy(unitcell, supercell_matrix=mul_matrix)  # Changed: use mul_matrix parameter
        phonon.generate_displacements(distance=0.02)

        # Get supercell information
        supercell = phonon.supercell
        sc_lattice = supercell.cell  # Changed: removed _params
        sc_a = np.linalg.norm(sc_lattice[0])
        sc_b = np.linalg.norm(sc_lattice[1])
        sc_c = np.linalg.norm(sc_lattice[2])
        n_atoms_super = len(supercell)
        
        print(f"Supercell lattice: a={sc_a:.3f}, b={sc_b:.3f}, c={sc_c:.3f} Å")
        print(f"Supercell atoms: {n_atoms_super}")
        print(f"Supercell volume: {supercell.volume:.4f} Å³")  # Added for info
        
        # Calculate forces
        supercells = phonon.supercells_with_displacements
        sets_of_forces = []
        
        print(f"Calculating {len(supercells)} displaced structures...")
        for j, scell in enumerate(supercells):
            atoms_disp = Atoms(
                symbols=scell.symbols,
                positions=scell.positions,
                cell=scell.cell,
                pbc=True
            )
            atoms_disp.calc = alignn_cal
            forces = atoms_disp.get_forces()
            sets_of_forces.append(forces)
        
        # Set forces and produce force constants
        phonon.forces = sets_of_forces
        phonon.produce_force_constants()
        
        # Calculate thermal properties
        phonon.run_mesh([50, 50, 50])
        phonon.run_thermal_properties(t_step=10, t_max=1500, t_min=0)
        phonon.write_yaml_thermal_properties(filename=f"thermal_properties.yaml-{i:02d}")
        print(f"Saved thermal_properties.yaml-{i:02d}")
    
    yaml_files = sorted(glob.glob("thermal_properties.yaml-*"))
    volumes, energies, yaml_files = _select_eos_basin(volumes, energies, yaml_files)

    np.savetxt("e-v.dat", np.column_stack([volumes, energies]), 
               fmt="%.8f", header="# volume(Å³)  energy(eV)")
    print("\nSaved e-v.dat")
    print(f"EOS points kept for QHA: {len(volumes)}  V=[{volumes.min():.2f},{volumes.max():.2f}]")
    
    if not yaml_files:
        raise FileNotFoundError("No thermal_properties.yaml-* written; cannot run phonopy-qha")

    phonopy_qha_path = None
    for candidate in (
        f"/home/a.burov/micromamba/envs/{env_used}/bin/phonopy-qha",
        "/home/a.burov/micromamba/envs/msdb/bin/phonopy-qha",
        shutil.which("phonopy-qha"),
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            phonopy_qha_path = candidate
            break
    if phonopy_qha_path is None:
        raise FileNotFoundError(
            f"phonopy-qha not found in {env_used} or msdb. "
            f"Install phonopy in {env_used} or msdb."
        )

    qha_env = os.environ.copy()
    qha_env["MPLBACKEND"] = "Agg"
    yaml_list = " ".join(yaml_files)
    cmd = (
        f"{phonopy_qha_path} -s --tmax=1000 --cutoff-frequency 0.1 "
        f"e-v.dat {yaml_list}"
    )
    print(f"\nRunning phonopy-qha from: {phonopy_qha_path}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=qha_env)
    if result.returncode != 0 and (
        "unrecognized arguments" in (result.stderr or "")
        or "invalid option" in (result.stderr or "").lower()
    ):
        print("phonopy-qha does not support --cutoff-frequency; retrying without it")
        cmd = f"{phonopy_qha_path} -s --tmax=1000 e-v.dat {yaml_list}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=qha_env)

    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"phonopy-qha failed:\n{result.stderr or result.stdout}")
    if not os.path.isfile("helmholtz-volume_fitted.dat"):
        raise FileNotFoundError(
            "phonopy-qha exited 0 but did not write helmholtz-volume_fitted.dat.\n"
            f"stderr:\n{result.stderr}"
        )
    print("QHA complete!")



# In[ ]:





# In[ ]:





# In[13]:


def visualize_data(potential, structure):
    """Helmholtz / α / G plots from phonopy-qha outputs."""
    fig_dir = _phonon_fig_dir()
    os.makedirs(fig_dir, exist_ok=True)

    qha_files = (
        "helmholtz-volume_fitted.dat",
        "thermal_expansion.dat",
        "gibbs-temperature.dat",
    )
    missing = [f for f in qha_files if not os.path.isfile(f)]
    if missing:
        print(
            "Skipping plots: phonopy-qha did not write "
            + ", ".join(missing)
            + ". Check the QHA log above."
        )
        return

    with open("helmholtz-volume_fitted.dat") as f:
        lines = f.readlines()

    fitted_start = None
    minima_start = None
    for i, line in enumerate(lines):
        if "# Fitted data" in line:
            fitted_start = i + 1
        if "# Minimas" in line:
            minima_start = i + 1
            break

    fitted_data = []
    for line in lines[fitted_start:minima_start - 1]:
        if line.strip() and not line.startswith("#"):
            fitted_data.append([float(x) for x in line.split()])
    fitted_data = np.array(fitted_data)
    volumes_fitted = fitted_data[:, 0]
    helmholtz_fitted = fitted_data[:, 1:]

    minima_data = []
    for line in lines[minima_start:]:
        if line.strip() and not line.startswith("#"):
            minima_data.append([float(x) for x in line.split()])
    minima_data = np.array(minima_data)
    vol_eq = minima_data[:, 0]
    energy_eq = minima_data[:, 1]

    n_temps = helmholtz_fitted.shape[1]
    print(f"Number of temperatures: {n_temps}")
    print(f"Number of volume points: {len(volumes_fitted)}")
    print(f"Equilibrium points: {len(vol_eq)}")

    fig, ax = plt.subplots(figsize=(8, 10))
    N = 5
    for i in range(n_temps):
        ax.plot(volumes_fitted[::N], helmholtz_fitted[::N, i], "o-", color="blue",
                linewidth=1.5, markersize=4, alpha=0.8)
    ax.plot(vol_eq, energy_eq, "*-", color="red", linewidth=1.5, markersize=10, zorder=10)
    ax.set_xlabel("Volume (Å³)", fontsize=fontsize)
    ax.set_ylabel("Free energy (eV)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{potential}_free_{structure}.png", dpi=450)
    plt.savefig(f"{fig_dir}/{potential}_free_{structure}.pdf", dpi=450)
    plt.close()
    print(f"Plot saved to: {fig_dir}/{potential}_free_{structure}")
    print(f"\nEquilibrium volumes:")
    print(f"  At T=0 K:   {vol_eq[0]:.4f} Å³")
    print(f"  At T={n_temps-1}00 K: {vol_eq[-1]:.4f} Å³")
    print(f"  Expansion: {vol_eq[-1] - vol_eq[0]:.4f} Å³")

    data = np.loadtxt("thermal_expansion.dat")
    temperatures, alpha = data[:, 0], data[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temperatures, alpha, "-", color="red", linewidth=2.5)
    ax.set_xlabel("Temperature (K)", fontsize=fontsize)
    ax.set_ylabel(r"Thermal expansion (K$^{-1}$)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(fontsize - 2)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{potential}_thermal_{structure}.pdf", dpi=450)
    plt.savefig(f"{fig_dir}/{potential}_thermal_{structure}.png", dpi=450)
    plt.close()
    print("✓ Thermal expansion coefficient plot saved")
    print(f"\nα at 0 K: {alpha[0]:.2e} ")
    print(f"α at 300 K: {alpha[30]:.2e}")
    print(f"α at 1000 K: {alpha[100]:.2e}")

    data = np.loadtxt("gibbs-temperature.dat")
    temperatures, gibbs = data[:, 0], data[:, 1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temperatures, gibbs, "-", color="red", linewidth=2.5)
    ax.set_xlabel("Temperature (K)", fontsize=fontsize)
    ax.set_ylabel("Gibbs free energy (eV)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 2)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{potential}_gibbs_{structure}.png", dpi=450)
    plt.savefig(f"{fig_dir}/{potential}_gibbs_{structure}.pdf", dpi=450)
    plt.close()
    print("✓ Gibbs free energy plot saved")
    print(f"\nG at 0 K:    {gibbs[0]:.2f} eV")
    print(f"G at 300 K:  {gibbs[30]:.2f} eV")
    print(f"G at 1000 K: {gibbs[100]:.2f} eV")



# In[ ]:





# In[14]:


def save_files(path_save):
    # create destination directory
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    # Define patterns to match
    patterns = [
        "*.pdf",
        "*.dat", 
        "*.png",
        "*.log",
        "*.traj",
        "thermal_properties.yaml*"
    ]
    
    # Move all matching files
    moved_count = 0
    for pattern in patterns: 
        if (pattern != f"{potential}.log"):
            files = glob.glob(pattern)
            for file in files:
                dest_path = os.path.join(path_save, os.path.basename(file))
                
                # Remove destination file if it exists
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                
                shutil.move(file, dest_path)  # Changed: use dest_path instead of path_save
                # print(f"Moved: {file} -> {dest_path}")
                moved_count += 1
    
    print(f"\n✓ Total files moved: {moved_count}")


# In[ ]:





# In[15]:


# Load equilibrium structure
# equilibrium_atoms = read("/home/a.burov/icys_2025/niohf/optimized/dft/delta.cif")


# In[16]:


# equilibrium_atoms = bulk('Al', 'fcc', a=4.05)



# In[ ]:





# In[18]:


# Rerun only gamma: previous QHA kept 4 E(V) points after basin cut and G(T)
# jumped by ~0.55 eV near 780 K. Compressed bowl is V≈150–158 (scale≈0.94–0.99);
# avoid the reconstructive expanded well above ~1.02.
structures_phases = [
    "gamma",
]


# In[ ]:


for st in structures_phases:
    if "layered" in st:
        mul_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        # p1/p21 sit on the compressed edge; p-1/p2c reconstruct when expanded.
        # Basin cut keeps one well; this window gives that well an interior minimum.
        # layered_p1 minimum sat at the compressed edge of [0.86, 1.14]
        scale = [0.76, 1.06]
        volume_points = 11
    elif st == "gamma":
        mul_matrix = [[4, 0, 0], [0, 3, 0], [0, 0, 1]]
        # Previous FIRE scan left this window with unconverged, uphill points.
        # Probe again on converged LBFGS energies and stay off the expanded well.
        scale = [0.88, 1.02]
        volume_points = 11
    else:
        mul_matrix = [[4, 0, 0], [0, 3, 0], [0, 0, 1]]
        # beta minimum was at the compressed edge of [0.90, 1.14]; alpha spikes above ~1.10
        scale = [0.86, 1.08]
        volume_points = 11

    path_file = f"{path_base}/{st}.cif"
    # read relaxed structures
    equilibrium_atoms = read(path_file)
    # equilibrium_atoms = bulk('Al', 'fcc', a=4.05)

    # path to save phonopy data
    path_data_phonopy = str(_UMLP_OUT / potential / st)

    # perform phonopy calculations
    try:
        calc_phonopy(
            equilibrium_atoms,
            env_used=env_used,
            mul_matrix=mul_matrix,
            scale=scale,
            volume_points=volume_points,
        )
    except RuntimeError as exc:
        print(f"{st}: skipped — {exc}")
        continue

    # visualize data and save plots (no-op if phonopy-qha did not write files)
    visualize_data(potential=potential, structure=st)

    # save phonopy files
    save_files(path_data_phonopy)

    print(f"{st}: finished without errors")


    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




