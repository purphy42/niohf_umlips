"""Shared DFT-matched phonon-dispersion paths and phonopy helpers.

Used by every UMLIP job under umlip_dispersion/. Structures, supercells,
displacement distance, and k-paths match phonons_dispersion.ipynb.

Cells are the already-relaxed DFT EOS supercells (P2₁/c 2×2×2, α 4×3×1);
DIM = 1 1 1 because every lattice vector is longer than 10 Å.

All plots go to umlip_dispersion/figures/ with the UMLIP name in the caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read, write
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

CLUSTER_ROOT = Path("/home/a.burov/icys_2025/niohf")
LOCAL_ROOT = Path("/home/arseniy/Desktop/work/niohf")
ROOT = CLUSTER_ROOT if CLUSTER_ROOT.exists() else LOCAL_ROOT
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

DISP_DISTANCE = 0.01
BAND_POINTS = 51

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

PHASES = {
    "layered_P2_1_c": {
        "structure": ROOT / "optimized_best/dft/layered_P2_1_c.vasp",
        "dim": IDENTITY,
        "paths": {
            "out_of_plane": {
                "labels": [r"$\Gamma$", "Z", "D", "B", r"$\Gamma$"],
                "qpoints": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.5],
                    [0.0, 0.0, 0.0],
                ],
            },
            "in_plane": {
                "labels": [r"$\Gamma$", "A", "E", "Z", r"C$_2$", r"Y$_2$", r"$\Gamma$"],
                "qpoints": [
                    [0.0, 0.0, 0.0],
                    [-0.5, 0.0, 0.5],
                    [-0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                    [-0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            },
        },
    },
    "diaspore_alpha": {
        "structure": ROOT / "optimized_best/dft/diaspore_alpha.vasp",
        "dim": IDENTITY,
        "paths": {
            "in_plane": {
                "labels": [r"$\Gamma$", "X", "S", "Y", r"$\Gamma$"],
                "qpoints": [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.0],
                ],
            },
            "out_of_plane": {
                "labels": [r"$\Gamma$", "Z", "U", "R", "T", "Z"],
                "qpoints": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.5],
                ],
            },
        },
        "vertical": {
            "labels": ["X", "U", "|", "Y", "T", "|", "S", "R"],
            "qpoints": [
                [0.5, 0.0, 0.0],
                [0.5, 0.0, 0.5],
                None,
                [0.0, 0.5, 0.0],
                [0.0, 0.5, 0.5],
                None,
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ],
        },
    },
}

PATH_ORDER = {
    "layered_P2_1_c": ["out_of_plane", "in_plane"],
    "diaspore_alpha": ["in_plane", "out_of_plane"],
}


def ase_to_phonopy(atoms: Atoms) -> PhonopyAtoms:
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=np.array(atoms.get_cell(), dtype=float),
        scaled_positions=np.array(atoms.get_scaled_positions(), dtype=float),
    )


def standardize_structure(atoms: Atoms) -> Atoms:
    """Seekpath primitive so BAND coords match the DFT notebook."""
    cell = (
        np.array(atoms.get_cell(), dtype=float),
        np.array(atoms.get_scaled_positions(), dtype=float),
        np.array(atoms.get_atomic_numbers(), dtype=int),
    )
    try:
        import seekpath

        res = seekpath.get_path(cell, with_time_reversal=True)
        std = Atoms(
            numbers=res["primitive_types"],
            cell=res["primitive_lattice"],
            scaled_positions=res["primitive_positions"],
            pbc=True,
        )
        print(
            "  seekpath primitive: "
            f"{len(std)} atoms, spacegroup {res.get('spacegroup_international')}"
        )
        return std
    except Exception as exc:
        print(f"  seekpath unavailable ({exc}); trying pymatgen KPathSeek")

    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.kpath import KPathSeek

        struct = AseAtomsAdaptor.get_structure(atoms)
        kpath = KPathSeek(struct)
        prim = kpath._prim  # standardized primitive
        std = AseAtomsAdaptor.get_atoms(prim)
        print(f"  pymatgen KPathSeek primitive: {len(std)} atoms")
        return std
    except Exception as exc:
        print(f"  no standardized primitive ({exc}); using input cell as-is")
        print("  WARNING: seekpath k-path coords may not match this cell orientation")
        return atoms


def path_segments(phase: str, include_vertical: bool = True):
    cfg = PHASES[phase]
    segs = []
    for key in PATH_ORDER[phase]:
        p = cfg["paths"][key]
        segs.append((key, p["labels"], np.array(p["qpoints"], dtype=float)))
    if include_vertical and "vertical" in cfg:
        raw_l = cfg["vertical"]["labels"]
        raw_q = cfg["vertical"]["qpoints"]
        cur_l, cur_q = [], []
        n = 0
        for lab, q in zip(raw_l, raw_q):
            if q is None or lab == "|":
                if len(cur_q) >= 2:
                    n += 1
                    segs.append((f"vertical_{n}", cur_l, np.array(cur_q, dtype=float)))
                cur_l, cur_q = [], []
                continue
            cur_l.append(lab)
            cur_q.append(q)
        if len(cur_q) >= 2:
            n += 1
            segs.append((f"vertical_{n}", cur_l, np.array(cur_q, dtype=float)))
    return segs


def build_band_path(phase: str, npoints: int = BAND_POINTS):
    bands, labels, path_connections = [], [], []
    for _, seg_labels, qs in path_segments(phase):
        nseg = len(qs) - 1
        if nseg < 1:
            raise ValueError(f"{phase}: path segment needs at least two q-points")
        for j in range(nseg):
            bands.append(np.linspace(qs[j], qs[j + 1], npoints))
        path_connections.extend([True] * (nseg - 1) + [False])
        labels.extend(seg_labels)
    n_groups = path_connections.count(False)
    expected = len(bands) + n_groups
    if len(labels) != expected:
        raise ValueError(
            f"{phase}: {len(labels)} labels vs {expected} expected "
            f"({len(bands)} segments, {n_groups} groups)"
        )
    return bands, labels, path_connections


def write_band_conf(phase: str, dest: Path) -> Path:
    dim = PHASES[phase]["dim"]
    lines = [
        "ATOM_NAME = Ni H O F",
        f"DIM = {dim[0][0]} {dim[1][1]} {dim[2][2]}",
        "PRIMITIVE_AXES = AUTO",
        f"BAND_POINTS = {BAND_POINTS}",
    ]
    all_labels = []
    for name, labels, qs in path_segments(phase):
        flat = "  ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in qs)
        lines.append(f"# {name}")
        lines.append(f"BAND = {flat}")
        if all_labels:
            all_labels.append("|")
        all_labels.extend(labels)
    token = []
    for lab in all_labels:
        token.append("G" if lab in (r"$\Gamma$", "G") else lab.replace("$", "").replace("\\", ""))
    lines.append("BAND_LABELS = " + " ".join(token))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n")
    return dest


def _band_xticks(distances, path_connections, labels):
    xticks, xlabels = [], []
    lab_i = 0
    for i, dist in enumerate(distances):
        new_group = i == 0 or not path_connections[i - 1]
        if new_group:
            xticks.append(float(dist[0]))
            xlabels.append(labels[lab_i] if lab_i < len(labels) else "")
            lab_i += 1
        xticks.append(float(dist[-1]))
        xlabels.append(labels[lab_i] if lab_i < len(labels) else "")
        lab_i += 1
    return xticks, xlabels


def plot_and_save_bands(phonon: Phonopy, potential: str, phase: str) -> None:
    """Single shared figures/ directory; UMLIP name in title and caption."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    fig, ax = plt.subplots(figsize=(10, 5.4))
    plotted = False
    try:
        data = phonon.get_band_structure_dict()
        distances = data["distances"]
        frequencies = data["frequencies"]
        _, labels, path_connections = build_band_path(phase)
        for dist, freq in zip(distances, frequencies):
            ax.plot(dist, freq, color="#1f4e79", lw=0.85)
        xticks, xlabels = _band_xticks(distances, path_connections, labels)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_xlim(xticks[0], xticks[-1])
        plotted = True
    except Exception as exc:
        print(f"  custom band plot failed ({exc}); falling back to phonopy.plot")
        plt.close(fig)
        phonon.plot_band_structure()
        fig = plt.gcf()
        ax = fig.axes[0] if fig.axes else None

    if ax is not None:
        ax.axhline(0.0, color="0.55", lw=0.6)
        ax.set_ylabel("Frequency (THz)")
        ax.set_title(f"{potential} — {phase}", fontsize=13, pad=8)

    fig.set_size_inches(10, 5.4)
    fig.text(
        0.5,
        0.01,
        f"uMLIP: {potential}",
        ha="center",
        fontsize=11,
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    stem = FIGURES_DIR / f"{potential}_{phase}_bands"
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close("all")
    print(f"  saved {stem}.png")


def forces_from_ase_calc(atoms: Atoms, calc) -> np.ndarray:
    atoms = atoms.copy()
    atoms.pbc = True
    atoms.calc = calc
    forces = np.asarray(atoms.get_forces(), dtype=float)
    if forces.ndim != 2 or forces.shape[1] != 3:
        raise ValueError(f"Unexpected force array shape {forces.shape}")
    return forces


def _assign_forces(phonon: Phonopy, sets_of_forces) -> None:
    n_disp = len(phonon.supercells_with_displacements)
    if len(sets_of_forces) != n_disp:
        raise ValueError(
            f"Got {len(sets_of_forces)} force sets for {n_disp} displacements"
        )
    if hasattr(phonon, "set_forces"):
        phonon.set_forces(sets_of_forces)
    else:
        phonon.forces = sets_of_forces


def _produce_force_constants(phonon: Phonopy) -> None:
    if hasattr(phonon, "produce_force_constants"):
        phonon.produce_force_constants()
    elif hasattr(phonon, "calculate_force_constants"):
        phonon.calculate_force_constants()
    else:
        raise RuntimeError("This phonopy version cannot produce force constants")


def _write_force_sets(phonon: Phonopy, dest: Path) -> None:
    try:
        from phonopy.file_IO import write_FORCE_SETS

        write_FORCE_SETS(phonon.dataset, filename=str(dest))
    except TypeError:
        from phonopy.file_IO import write_FORCE_SETS

        write_FORCE_SETS(phonon.dataset)
        Path("FORCE_SETS").replace(dest)
    except Exception as exc:
        print(f"  FORCE_SETS not written ({exc})")


def run_dispersion(potential: str, force_fn, job_dir: Path) -> None:
    """Compute DFT-matched bands for layered P2₁/c and diaspore α."""
    job_dir = Path(job_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for phase, cfg in PHASES.items():
        work = job_dir / phase
        work.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {potential} / {phase} ===")
        print(f"  structure: {cfg['structure']}")
        print(f"  dim: {cfg['dim']}")
        print(f"  figures: {FIGURES_DIR}")

        if not cfg["structure"].exists():
            raise FileNotFoundError(f"Missing DFT reference: {cfg['structure']}")

        atoms = read(str(cfg["structure"]))
        write(str(work / "POSCAR"), atoms, format="vasp", direct=True)
        write_band_conf(phase, work / "band.conf")

        phonon = Phonopy(
            ase_to_phonopy(atoms),
            supercell_matrix=cfg["dim"],
            primitive_matrix="auto",
        )
        phonon.generate_displacements(distance=DISP_DISTANCE)
        disps = list(phonon.supercells_with_displacements)
        if any(sc is None for sc in disps):
            raise RuntimeError(
                f"{phase}: phonopy returned a None supercell; "
                "cannot align forces with displacements"
            )
        print(f"  displacements: {len(disps)}  (d={DISP_DISTANCE} Å)")

        sets_of_forces = []
        for i, scell in enumerate(disps, start=1):
            atoms_disp = Atoms(
                symbols=list(scell.symbols),
                positions=np.array(scell.positions, dtype=float),
                cell=np.array(scell.cell, dtype=float),
                pbc=True,
            )
            forces = np.asarray(force_fn(atoms_disp), dtype=float)
            if forces.shape != (len(scell.symbols), 3):
                raise ValueError(f"Bad force shape {forces.shape} at disp {i}")
            if not np.isfinite(forces).all():
                raise ValueError(f"Non-finite forces at disp {i}")
            sets_of_forces.append(forces)
            print(f"    disp {i}/{len(disps)}")

        _assign_forces(phonon, sets_of_forces)
        _produce_force_constants(phonon)
        _write_force_sets(phonon, work / "FORCE_SETS")
        try:
            phonon.save(str(work / "phonopy.yaml"), settings={"force_constants": True})
        except TypeError:
            phonon.save(str(work / "phonopy.yaml"))

        bands, labels, path_connections = build_band_path(phase)
        try:
            phonon.run_band_structure(
                bands,
                is_band_connection=True,
                path_connections=path_connections,
                labels=labels,
            )
        except TypeError:
            phonon.run_band_structure(bands, path_connections=path_connections, labels=labels)
        except Exception as exc:
            print(f"  band connection failed ({exc}); retrying without it")
            phonon.run_band_structure(
                bands,
                is_band_connection=False,
                path_connections=path_connections,
                labels=labels,
            )

        try:
            phonon.write_yaml_band_structure(filename=str(work / "band.yaml"))
        except TypeError:
            phonon.write_yaml_band_structure()
            Path("band.yaml").replace(work / "band.yaml")

        plot_and_save_bands(phonon, potential, phase)

    print(f"\nFinished {potential} without errors")
