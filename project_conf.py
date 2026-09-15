# -*- coding: utf-8 -*-
"""
User-related parameters for siman, file is installed to home folder

"""
from __future__ import division, unicode_literals, absolute_import 
from siman.header import CLUSTERS

"""Cluster constants"""
DEFAULT_CLUSTER = 'razor128' #short name of cluster
PATH2ARCHIVE = '' # path to archive; if no files are found at home folder, siman will check here; relative paths should be same
CLUSTERS = {}
PATH2PROJECT = 'icys_2025/niohf' # path to project on cluster relative to home folder
PATH_TO_PROJECT_ON_COMP = "/home/a.burov/soft/vasp_potentials/potpaw_PBE_MPIE/"


user = 'a.burov'
# Cluster settings
CLUSTERS['razor128'] = {
'address':'razor128',
'vasp_com':'srun vasp_gam',
'homepath':'/home/a.burov/',
'schedule':'SLURM',
'corenum':16,
'modules': 'source /etc/profile.d/modules.sh; \
module load devtools/mpi/mpich/4.2.1/gcc/11.2; \
module load q-ch/qe/7.3.1/gcc/11.2/mpich/mkl; \
module load q-ch/vasp/5.4.4_mpich_mkl; \
\nulimit -s unlimited\n\
'
}

CLUSTERS['razor64'] = {
'address':'razor64',
'vasp_com':'mpirun -np 8 vasp_std',
'homepath':'/home/a.burov/',
'schedule':'SLURM',
'corenum': 8,
'procmemgb': 4,
'modules': 'source /etc/profile.d/modules.sh; \
module load devtools/compiler/aocl/4.0.0; \
module load devtools/mpi/openmpi/4.1.5/gcc/11.3; \
module load q-ch/vasp/6.4.3; \
\nulimit -s unlimited\n\
'
}

# One SLURM job per displacement (Gamma-only supercells)
CLUSTERS['zen4'] = {
'address': 'razor128',
'homepath': '/home/a.burov/',
'schedule': 'SLURM',
'corenum': 16,
'partition': 'zen4',
'walltime': '24:00:00',
'any_commands': ['--mem=64G'],
'vasp_com': 'srun vasp_gam',
'modules': 'source /etc/profile.d/modules.sh; \
module load zen4/vasp/5.4.4_openmpi_gcc_aocl; \
ulimit -s unlimited\n\
export OMP_NUM_THREADS=1\n',
}


"""Local constants"""
PATH2POTENTIALS = '/home/a.burov/potpaw_paw/potpaw_PBE_MPIE/'
PATH2JMOL = 'java -jar Jmol.jar'
PATH2NEBMAKE = '~/tools/vts/nebmake.py'
pmgkey = "AWqKPyV8EmTRlf1t" #MAPI_KEY can be generated in the following webpage: https://materialsproject.org/dashboard 
EXCLUDE_NODES = False

cluster_tools = '/home/a.burov/tools/'
show_head = None # show header for res_loop()

pmgkey = "dzpOVLsVP3VWa3VIOB"
mpkey = "LTSM6dStBrl69FjopxP7KdZBP35B1yh7"

