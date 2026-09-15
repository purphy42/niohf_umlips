
import copy
# from header import * 
from siman import header
from siman.set_functions import InputSet, inherit_iset, make_sets_for_conv, init_default_sets
"""
remove update_set() completly
"""

# siman in msdb initializes InputSet.setup as {'timestep'} (a Python set).
# InputSet.update() then does self.setup['timestep'] = POTIM and crashes.
if not getattr(InputSet, "_niohf_setup_patched", False):
    _siman_init = InputSet.__init__
    _siman_update = InputSet.update

    def _init_setup_dict(self, *args, **kwargs):
        _siman_init(self, *args, **kwargs)
        if not isinstance(getattr(self, "setup", None), dict):
            self.setup = {}

    def _update_setup_dict(self, *args, **kwargs):
        if not isinstance(getattr(self, "setup", None), dict):
            self.setup = {}
        return _siman_update(self, *args, **kwargs)

    InputSet.__init__ = _init_setup_dict
    InputSet.update = _update_setup_dict
    InputSet._niohf_setup_patched = True



#1 - only volume; v  isif = 5
#2 - full relax;    vsa
#8 - no relaxation; 0
#9 - only atoms;    a


#set_potential uses relative paths







"""List of user VASP sets obtained on inheritance principle """
"""Syntax:  ("set_new", "set_old", {"param1":value1, "param2":value2, ...})
        - set_new - name of new set
        - set_old - name of base set used for creating new set
        - {} -      dictionary of parameters to be updated in set_new

"""
  # Ueff::list(string=numeric):=["V"=3.1,"Cr"=3.5,"Mn"=3.9,"Fe"=4.0,"Co"=3.4,"Ni"=6.0,"Cu"=4.0,"Mo"=3.5,"Ag"=1.5]
dftu_packet = {'ISTART'   :1,   'ICHARG':1,  'LDAUTYPE':2, 'LASPH':'.TRUE.', 
                'LDAUPRINT':2, 'LMAXMIX' :4, 'LDAU' :'.TRUE.',
                'LDAUL':{'Ti':2,   'Co':2  , 'Fe':2  , 'Ni':2  , 'Mn':2  , 'V':2   , 'Cr':2 },
                'LDAUU':{'Ti':0,   'Co':3.4, 'Fe':4.0, 'Ni':6.2, 'Mn':3.9, 'V':3.1 , 'Cr':3.5, 'Fe/S':1.9 },
                'LDAUJ':{'Ti':0.0, 'Co':0.0, 'Fe':0.0, 'Ni':0.0, 'Mn':0.0, 'V':0.0 , 'Cr':0.0, 'Fe/S':0   } } # universal set, Jain2011 azh values, Ni from genome
dftu_packet1 = {'ISTART'   :1,   'ICHARG':1,  'LDAUTYPE':2, 'LASPH':'.TRUE.', 
                'LDAUPRINT':2, 'LMAXMIX' :4, 'LDAU' :'.TRUE.',
                'LDAUL':{'Ti':2,   'Co':2  , 'Fe':2  , 'Ni':2  , 'Mn':2  , 'V':2   , 'Cr':2, 'W':2 },
                'LDAUU':{'Ti':0,   'Co':3.4, 'Fe':4.0, 'Ni':6.2, 'Mn':3.9, 'V':3.1 , 'Cr':3.5, 'Fe/S':1.9, 'W':4 },
                'LDAUJ':{'Ti':0.0, 'Co':0.0, 'Fe':0.0, 'Ni':0.0, 'Mn':0.0, 'V':0.0 , 'Cr':0.0, 'Fe/S':0, 'W':1   } }
dftu_packet_h = {'ISTART'   :1,   'ICHARG':1,  'LDAUTYPE':2, 'LASPH':'.TRUE.', 
                'LDAUPRINT':2, 'LMAXMIX' :4, 'LDAU' :'.TRUE.', 
                'LDAUL':{'Ti':2,   'Co':2  , 'Fe':2  , 'Ni':2  , 'Mn':2  , 'V':2   , 'Cr':2,               'W':2 },
                'LDAUU':{'Ti':0,   'Co':3.4, 'Fe':4.0, 'Ni':6.2, 'Mn':5,   'V':3.1 , 'Cr':3.5, 'Fe/S':1.9, 'W':4 },
                'LDAUJ':{'Ti':0.0, 'Co':0.0, 'Fe':0.0, 'Ni':0.0, 'Mn':0.0, 'V':0.0 , 'Cr':0.0, 'Fe/S':0,   'W':1   } }

dftu_packet_h2 = {'ISTART'   :1,   'ICHARG':1,  'LDAUTYPE':2, 'LASPH':'.TRUE.', 
                'LDAUPRINT':2, 'LMAXMIX' :4, 'LDAU' :'.TRUE.',
                'LDAUL':{'Ti':2,   'Co':2  , 'Fe':2  , 'Ni':2  , 'Mn':2  , 'V':2   , 'Cr':2,               'W':2 , 'Ru': 2  , 'Mo': 2},
                'LDAUU':{'Ti':0,   'Co':5,   'Fe':4.0, 'Ni':6.2, 'Mn':5,   'V':3.1 , 'Cr':3.5, 'Fe/S':1.9, 'W':4 , 'Ru': 4  , 'Mo': 3},
                'LDAUJ':{'Ti':0.0, 'Co':0.0, 'Fe':0.0, 'Ni':0.0, 'Mn':0.0, 'V':0.0 , 'Cr':0.0, 'Fe/S':0,   'W':1 , 'Ru': 0.0, 'Mo': 0} }

dftu_packet_h3 = {'ISTART'   :1,   'ICHARG':1,  'LDAUTYPE':2, 'LASPH':'.TRUE.', 
                'LDAUPRINT':2, 'LMAXMIX' :4, 'LDAU' :'.TRUE.',
                'LDAUL':{'Ti':2,   'Co':2  , 'Fe':2  , 'Ni':2  , 'Mn':2  , 'V':2   , 'Cr':2,               'W':2 , 'Ru': 2  , 'Mo': 2, 'Zn':2},
                'LDAUU':{'Ti':0,   'Co':5,   'Fe':4.0, 'Ni':6.2, 'Mn':5,   'V':3.1 , 'Cr':3.5, 'Fe/S':1.9, 'wW':4 , 'Ru': 4  , 'Mo': 3, 'Zn':5},
                'LDAUJ':{'Ti':0.0, 'Co':0.0, 'Fe':0.0, 'Ni':0.0, 'Mn':0.0, 'V':0.0 , 'Cr':0.0, 'Fe/S':0,   'W':1 , 'Ru': 0.0, 'Mo': 0, 'Zn':0} }


dftu_packet_rsf = {'ISTART'   :1,   'ICHARG':1,  'LDAUTYPE':2, 'LASPH':'.TRUE.',
                'LDAUPRINT':2, 'LMAXMIX' :4, 'LDAU' :'.TRUE.',
                'LDAUL':{'Ti':2,   'Co':2  , 'Fe':2  , 'Ni':2  , 'Mn':2  , 'V':2   , 'Cr':2,               'W':2 , 'Ru': 2  , 'Mo': 2, 'Zn':2},
                'LDAUU':{'Ti':0,   'Co':3.32,   'Fe':4.0, 'Ni':6.2, 'Mn':3.9,   'V':3.25 , 'Cr':3.7, 'Fe':5.3, 'W':6.2, 'Ru': 4  , 'Mo': 4.38, 'Zn':5},
                'LDAUJ':{'Ti':0.0, 'Co':0.0, 'Fe':0.0, 'Ni':0.0, 'Mn':0.0, 'V':0.0 , 'Cr':0.0, 'Fe':0,   'W':1 , 'Ru': 0.0, 'Mo': 0, 'Zn':0} }

pot_pack_rsf = {'set_potential':{1: "H", 2: "He", 3:"Li_sv", 4: "Be_sv", 5: "B", 6: "C", 7: "N",  8:"O", 9:"F", 10: "Ne",  11:'Na_pv',
    12: "Mg_pv", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K_sv", 20: "Ca_sv", 21: "Sc_sv", 22: "Ti_pv", 23:"V_pv", 
    24: "Cr_pv", 25: "Mn_pv", 26: "Fe_pv", 27: "Co", 28: "Ni_pv", 29: "Cu_pv", 30: "Zn", 31: "Ga_d", 32: "Ge_d", 33: 'As', 34: "Se",
    35: "Br", 36: "Kr", 37: "Rb_sv", 38: "Sr_sv", 39: "Y_sv", 40: "Zr_sv", 41: "Nb_pv", 42: "Mo_pv", 43: "Tc_pv", 44: "Ru_pv", 45: "Rh_pv", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In_d", 50: "Sn_d", 51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs_sv", 56: "Ba_sv", 57: "La", 
    58: "Ce", 59: "Pr_3", 60: "Nd_3", 61: "Pm_3", 62: "Sm_3", 63: "Eu", 64: "Gd", 65: "Tb_3", 66: "Dy_3", 67: "Ho_3", 68: "Er_3", 
    69: "Tm_3", 70: "Yb_3", 71: "Lu_3", 72: "Hf_pv", 73: "Ta_pv", 74: "W_pv", 75: "Re_pv", 76: "Os_pv", 77: "Ir",  78: "Pt", 79: "Au", 
    80: "Hg", 81: "Tl", 82: "Pb_d", 83: "Bi", 84: "Po", 85: "At_d", 86: "Rn", }} #except O_sv, which requires 1000 eV ecut at least


dftu_packet_off = {'LDAU' :None, 'LASPH':None, 'LDAUPRINT':None, 'LDAUTYPE':None,  'LDAUL':None, 'LDAUU':None, 'LDAUJ':None, }

YBaCoO_dftu =  {'LDAUU':{'Co':5}, 'LDAUJ':{'Co':0.8} }

mag_packet = {
    'GGA_COMPAT': '.FALSE.',
    'ISPIN':2,
    'LORBIT':11, #more info
    'magnetic_moments':{'Ti':0.6, 'V':5, 'Fe':5, 'Co':5, 'Mn':5, 'Ni':5, 'Cr':5 }

}

mag_packet_n = {

    'magnetic_moments':{'Ti':0.6, 'V':5, 'Fe':5, 'Co':0, 'Mn':5, 'Ni':5, 'Cr':5, 'Mo':0 }
}

mag_packet_lfp = {

    'magnetic_moments':{'Fe':5, 'Co':4, 'Mn':5, 'Ni':3 }
}


#hybrid packet
hse6_pack = {'ISTART':1, 'LHFCALC':'.TRUE.', 'HFSCREEN':0.2, 'add_nbands':1.1, 'ALGO':'All', 'TIME':0.4}
hse6_pack_low = hse6_pack.copy()
hse6_pack_low.update({'PRECFOCK':'Fast', 'NKRED':2})


mix_mag_packet =  {'AMIX':0.2, 'BMIX':0.00001, 'AMIX_MAG':0.8, 'BMIX_MAG':0.00001} #linear mixing fine
mix_mag_packet_fine =  {'AMIX':0.1, 'BMIX':0.00001, 'AMIX_MAG':0.4, 'BMIX_MAG':0.00001} #even finer linear mixing

ion_relax_packet = {'NSW':25, 'EDIFFG':-0.025, 'EDIFF':0.0001, 'ISIF':2}
static_run_packet = {'NSW':0, 'EDIFF'     : 6e-06, 'NELM':50}
my_low_pack = {'KSPACING':0.3, 'ENCUT':400, 'ENAUG':400*1.75, 'POTIM':0.2, 'NELM':20, 'EDIFFG':-0.05 }
acc_pack  = {'PREC':'Accurate', 'ADDGRID':'.TRUE.', 'EDIFF':6e-6, 'EDIFFG':-0.010, 'NELM':50, 'NSW':50, 'ISTART':1, 'ICHARG':0 }
acc_pack_relax  = {'PREC':'Accurate', 'ADDGRID':'.TRUE.', 'EDIFF':1e-8, 'EDIFFG':-0.010, 'NSW':50, 'NELM':50, 'ISTART':1, 'ICHARG':0 }
acc_pack2_stat   = {'PREC':'Accurate', 'ADDGRID':'.TRUE.', 'EDIFF':1e-8, 'LREAL':False, 'NSW':0, 'NELM':50, 'ISTART':1, 'ICHARG':0 }
acc_pack2_relax  = {'PREC':'Accurate', 'ADDGRID':'.TRUE.', 'EDIFF':1e-8, 'LREAL':False, 'EDIFFG':-0.025, 'NELM':50, 'NSW':50, 'ISTART':1, 'ICHARG':0 }
acc_pack4_relax  = {'PREC':'Accurate', 'ADDGRID':'.TRUE.', 'EDIFF':1e-4, 'LREAL':False, 'EDIFFG':-0.025, 'NELM':50, 'NSW':50, 'ISTART':1, 'ICHARG':0 }


dos_pack = {'NSW':0, 'LORBIT':12, 'ISMEAR':-5, 'SIGMA':None, 'LAECHG':'.TRUE.', 'EMIN':-10, 'EMAX':14, 'NEDOS':2000, 'KSPACING':0.15, 'savefile':'dox'}
bader_pack = {'PREC':'Accurate', 'ADDGRID':'.TRUE.', 'EDIFF':1e-08, 'LAECHG':'.TRUE.', 'NELM':100, 'NSW':0, 'ICHARG':1, 'savefile' : 'acox'}
surface_pack = {'AMIN':0.01, 'AMIX':0.2, 'BMIX':0.001, 'NELMIN':8, 'IDIPOL':3, 'LDIPOL':'.TRUE.', 'LVTOT':'.TRUE.'} # from pymatgen
surface_pack2 = {'AMIN':None, 'AMIX':None, 'BMIX':None, 'NELMIN':8, 'IDIPOL':3, 'LDIPOL':'.TRUE.', 'LVTOT':'.TRUE.', 'NELM':50, 'ICHARG':1} # 
# slab_incar["DIPOL"] = structure.center_of_mass # please consider


partial_chg_pack = {'savefile':'p', 'LWAVE':False, 'ICHARG':0, 'LPARD':'TRUE', 'EINT':'-0.75 0', 'NBMOD':-3}


dos_pack2 = dos_pack.copy()
dos_pack3 = dos_pack.copy()
dos_pack3.update({'PREC':'Accurate', 'ADDGRID':'.TRUE.', 'EDIFF':6e-6, 'NELM':50, })
# dos_pack2.update({83:'Bi_pv', 34:'Se'})
mag_relax = mag_packet.copy()
mag_relax.update(ion_relax_packet)
# print mag_relax
sv_pot_pack = {'set_potential':{3:"Li_sv2",    8:"O", 9:"F", 11:'Na_sv', 37:'Rb_sv', 15:"P", 16:'S', 19:'K_sv', 22:"Ti_sv_new", 23:"V_sv_new", 25:"Mn_sv",    26:"Fe_sv",     27:"Co_sv" , 28:"Ni_pv", 33:'As_d'  }} #except O_sv, which requires 1000 eV ecut at least
sv_pot_pack_sn = {'set_potential':{3:"Li_sv2",    8:"O", 9:"F", 11:'Na_sv', 37:'Rb_sv', 15:"P", 16:'S', 19:'K_sv', 22:"Ti_sv_new", 23:"V_sv_new", 25:"Mn_sv",    26:"Fe_sv",     27:"Co_sv" , 28:"Ni_pv", 33:'As_d' , 50:"Sn_d" }} #except O_sv, which requires 1000 eV ecut at least
pot_pack = {'set_potential':{1:'H', 3:"Li",  5:'B', 6:'C',  8:"O", 9:"F", 11:'Na', 12:'Mg', 15:"P", 16:'S', 19:'K_pv',     20:'Ca',         22:"Ti",        23:"V", 24:'Cr',   25:"Mn",       26:"Fe",        27:"Co_new", 28:"Ni_new", 33:'As', 37:'Rb_pv', 39:'Y_sv', 45:'Rh', 56:'Ba_sv',   83:'Bi_pv', 34:'Se',    }  }
over = ''

YBC8mag2 = '0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6'
YBC8mag4 = '0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 -5 -5 5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6 0.6'

band_pack_monibc = {'ICHARG':11, 'LORBIT':11, 
             'k_band_structure':[101, ('G', 0, 0, 0), ('R', 0.5, 0.5, 0.5), 
                                 ('R\'', -0.5, -0.5, -0.5), ('T', 0, 0.5, 0.5), ('T\'', 0, -0.5, -0.5),
                                 ('U', 0.5, 0, 0.5), ('U\'', -0.5, 0, -0.5), ('V', 0.5, 0.5, 0),
                                 ('V\'', -0.5, -0.5, 0), ('X', 0.5, 0, 0), ('X\'', -0.5, 0, 0),
                                 ('Y', 0, 0.5, 0), ('Y\'', 0, -0.5, 0), ('Z', 0, 0, 0.5), ('Z\'', 0,0,-0.5)] }

user_vasp_sets = [
# ('8sv','8',{'set_potential':{3:"Li_sv2",    8:"O", 9:"F", 11:'Na_sv', 15:"P", 16:'S', 19:'K_sv', 22:"Ti_sv_new", 23:"V_sv_new", 25:"Mn_sv",    26:"Fe_sv",     27:"Co_sv" , 28:"Ni_pv", 33:'As_d'  }  }),
# ('8pv','8',{'set_potential':{3:"Li_sv2",    8:"O", 9:"F", 11:'Na_pv', 15:"P", 16:'S', 19:'K_pv', 22:"Ti_pv",     23:"V_pv",     25:"Mn_pv_new",26:"Fe_pv_new", 27:"Co_new", 28:"Ni_pv", 33:'As'  }  }),
# ('8', 'static', {}),
# ('8','8',pot_pack,),
# ('9',    '8', ion_relax_packet),

# ('9e3', '9', {'set_potential':{13:"Al"},'NSW':25, 'EDIFFG':-0.025, 'EDIFF':0.00001, 'ISIF':2, 'ENCUT':350, 'ENAUG': 525, 'add_nbands':1.2,}, over),


# ('9e3l', '9e3', {'NSW':25, 'EDIFFG':-0.05, 'EDIFF':0.001,  'ENCUT':300,  'KSPACING':0.6}, '0'),
# ('9eg3l', '9e3l', {'ISMEAR':0}, '0'),
# ('9e3ls', '9e3l', surface_pack2, '0'),


# ('9e3i2', '9e3', {'IBRION': 2, 'NSW': 45, 'NPAR':4, 'PREC' : 'Accurate'}, over),
# ('9e3is3', '9e3i2', {'ISIF': 3}, over),
# ('4e3n', '9e3i2', {'ISIF': 4, 'ISYM':-1}, over),
# ('8e3n', '4e3n', {'NSW': 0}, over),

# ('dos', '9e3i2', dos_pack),
# ('mag', '9e3i2', mag_packet),
# ('elastic3', 'mag', {'IBRION':6, 'ISIF':3, 'PREC': 'Accurate', 'NPAR':1, 'POTIM': 0.1}, ),
# ('elastic7', 'elastic3', {'ENCUT':700}, ),
# ('ph', 'mag', {'NSW':0}),



# ('elf1', 'mag', {'LELF': '.TRUE.', 'ISTART':1, 'NPAR': 1, 'NSW': 0, 'NSIM': None}),






('8', 'static', {}),
('8','8',pot_pack,),
('9',    '8', ion_relax_packet),
('8U',    '8', dftu_packet ),

('9u', '8u',    ion_relax_packet),  #ion relax 


('9bulk' , '9',  {'NSW':50,'EDIFFG':-0.05, 'EDIFF':1e-5, 'ISIF':2, 'ENCUT': 400, 'ENAUG': 600, 'add_nbands':1.3, 
    'KSPACING': 0.7, "LDAU": ".TRUE.", 'LDAUTYPE': 2, 'LDAUPRINT': 2, 'LDAUL':{'Li': -1, 'Na': -1, 'K': -1, 'Ni': 2, 'O': -1, 'H': -1, 'F': -1,}, 'LDAUU': {'Li': 0, 'Na': 0, 'K': 0, 'Ni': 6.2, 'O': 0, 'H': 0, 'F': 0}, 'ISMEAR': 0, 'SIGMA':0.1,'NELM':100, 'NPAR': None, 'LREAL': 'Auto', 'ISTART': 1,  'ISPIN': 2, 'LMAXMIX': 4, 'POTIM': 0.5, 'LASPH': '.TRUE.', 'LORBIT': 11,  'MAXMIX': None,  'GGA_COMPAT': ".FALSE.", 'PREC': 'Accurate', 'LPLANE': ".TRUE", 'LSCALU': '.FALSE.', 'ALGO': 'Normal',  "NELMIN": None, "IBRION": 1, 'LDAUJ': {'Li': 0, 'Na': 0, 'K': 0, 'Ni': 0.0, 'O': 0, 'H': 0.0, 'F': 0, "P": 0}, "LWAVE": ".FALSE.", "IWAVPR": 11, "POTIM": 0.15,   }, 'over'),

('9bulk_sp' , '9bulk',  {"IBRION": -1, "NSW": 0  }, 'over'),
('9bulk_fine' , '9bulk',  {"NSW": 50, "EDIFF": 1e-6, "KSPACING": 0.3, "EDIFFG": -0.01}, 'over'),
('9bulk_fine_vdw' , '9bulk',  {"NSW": 50, "EDIFF": 1e-6, "KSPACING": 0.3, "IVDW": 11, "EDIFFG": -0.01}, 'over'),
('9bulk_fine_rel' , '9bulk',  {"NSW": 50, "ENCUT": 600, "ENAUG": 900, "ISIF": 3, 
    "EDIFF": 1e-6, "KSPACING": 0.3, "EDIFFG": -0.01}, 'over'),
('9bulk_fine_rel_vdw' , '9bulk',  {"NSW": 50, "ENCUT": 600, "ENAUG": 900, "ISIF": 3, 
                        "EDIFF": 1e-6, "KSPACING": 0.3, "IVDW": 11, "EDIFFG": -0.01}, 'over'),

('9_bulk_mp', 'bulk', {'NSW':99, 'EDIFFG': -0.05, 'EDIFF': 1e-5, 'ISIF':2, 'ENCUT':520, 'ENAUG': 780, 'add_nbands': 1.2, "GGA_COMPAT": ".FALSE.", 'KSPACING': 0.3, 'ISMEAR': 0, 'SIGMA': 0.2, 'NELM': 100, 'NPAR': None, 'LREAL': 'Auto', 'ISTART': 1,  'ISPIN': 2, "LORBIT": 11, "LAECHG": ".TRUE.", 'POTIM': 0.2, 'LASPH': '.TRUE.',  'MAXMIX': None, 'GGA_COMPAT': None, 'PREC': 'Accurate', 'LPLANE': None, 'ALGO': 'Normal',  "NELMIN": None, "LWAVE": ".FALSE.", "IWAVPR": None,
    "IBRION": 1, "LMIXTAU": ".TRUE.", **pot_pack_rsf,}, 'over'),


('9bulk_eos' , '9bulk',  {"IVDW": 11, "ISIF": 4, "EDIFF": 1e-5, "NSW": 100, "POTIM": 0.15, **dftu_packet_off}, 'over'),

('phonons_prio' , '9bulk',  {"NSW": 150, "EDIFF": 1e-7, "EDIFFG": -0.001, "LREAL": ".FALSE", "LWAVE": ".FALSE", }, 'over'),
('phonons_ibrion6_400' , 'phonons_prio',  {"ISIF": 3, "NFREE": 4, "IBRION": 6, "EDIFF": 1e-7, "PREC": "Accurate", "NSW": 1}, 'over'),
('phonons_ibrion6_300' , 'phonons_ibrion6_400',  {"ENCUT": 300, "ENAUG": 450}, 'over'),
('phonons_ibrion6_500' , 'phonons_ibrion6_400',  {"ENCUT": 500, "ENAUG": 750}, 'over'),
('phonons_ibrion6_600' , 'phonons_ibrion6_400',  {"ENCUT": 600, "ENAUG": 900}, 'over'),
('phonons_ibrion6_700' , 'phonons_ibrion6_400',  {"ENCUT": 700, "ENAUG": 1050}, 'over'),
('phonons_ibrion6_800' , 'phonons_ibrion6_400',  {"ENCUT": 800, "ENAUG": 1200}, 'over'),
('phonons_ibrion6_900' , 'phonons_ibrion6_400',  {"ENCUT": 900, "ENAUG": 1350}, 'over'),
('phonons_ibrion6_1000' , 'phonons_ibrion6_400',  {"ENCUT": 1000, "ENAUG": 1500}, 'over'),


('phonons_ibrion6_ksp_1' , 'phonons_ibrion6_700',  {"KSPACING": 1.0}, 'over'),
('phonons_ibrion6_ksp_07' , 'phonons_ibrion6_700',  {"KSPACING": 0.6}, 'over'),
('phonons_ibrion6_ksp_035' , 'phonons_ibrion6_700',  {"KSPACING": 0.35}, 'over'),
('phonons_ibrion6_ksp_02' , 'phonons_ibrion6_700',  {"KSPACING": 0.201}, 'over'),
('phonons_ibrion6_ksp_015' , 'phonons_ibrion6_700',  {"KSPACING": 0.15}, 'over'),
('phonons_ibrion6_ksp_01' , 'phonons_ibrion6_700',  {"KSPACING": 0.1}, 'over'),

('phonons_setup_rel' , 'phonons_prio',  {"KSPACING": 0.7, "ENCUT": 700, "ENAUG": 1050}, 'over'),
('phonons_setup_eos' , 'phonons_setup_rel',  {"ISIF": 4, "LREAL": ".FALSE.", "NSW": 150 }, 'over'),
# Static forces on phonopy supercells (Gamma-only via add(..., ngkpt=[1,1,1]))
('phonons_disp' , 'phonons_setup_rel', {
    "IBRION": -1, "NSW": 0, "ISIF": 2, "EDIFF": 1e-8,
    "PREC": "Accurate", "ADDGRID": ".TRUE.", "LREAL": ".FALSE.",
    "LWAVE": ".FALSE.", "LCHARG": ".FALSE.", "savefile": "ox",
}, 'over'),

('1' ,'9', my_low_pack ),
('1m' ,'1', mag_packet ),
('0m' ,'1m',    static_run_packet, ),  #
('0mAB', '0m', bader_pack),
('0mABI0', '0mAB', {'ISMEAR':0, 'SIGMA':0.1}),
('0mbox', '0m',    {'magnetic_moments':{'O':1.}, 'ISTART':0,'KSPACING':10., 'KGAMMA':'.FALSE.','add_nbands':2, 'ISMEAR':0, 'LREAL':'.FALSE.', 'SIGMA':0.01},),  #
('0mboxn',    '0mbox', {'NELM':100, 'add_nbands':4}),


('1ur' ,'9u',   {'u_ramping_nstep':3, 'KSPACING':0.3, 'ENCUT':400, 'ENAUG':400*1.75, 'POTIM':0.2, 'NELM':20, 'EDIFFG':-0.05 } ),  # low quality
('1ur10' ,'1ur',    {'u_ramping_nstep':10 } ),  #
('1urk15' ,'1ur',    {'KSPACING':0.15 } ),  #
('1urp03' ,'1ur',    {'POTIM':0.03 } ),  #
('1urn1' ,'1ur',    {'NELM':10 } ),  #
('1urnp' ,'1ur',    {'NPAR':4 } ),  #
('1urNM' ,'1ur',    {'NELMIN':8 } ),  #
('1u' ,'1ur',    {'u_ramping_nstep':None }, ),  #
('1ui' ,'1u', {} ),  #for inherit_xred option, make control from here


('1u_co' , '1u',  {**mag_packet_n, 'NPAR': 4}, ''),
('1ul_new' ,'1u',     {'IDIPOL':3, 'LPLANE': '.FALSE.'}, ),
('1uc' ,'1u',     {'IBRION':2, 'LPLANE': '.FALSE.'}, ),
('1ulc' ,'1ul_new',     {'IBRION':2}, ),
('1ULC' ,'1ulc',     {'IBRION':2, 'PREC':'Accurate', 'NSW': 45, 'NPAR': 4}),
('1ULC_co' ,'1ulc',     {'IBRION':2, 'PREC':'Accurate', 'NSW': 45, 'NPAR': 4, 'MAGMOM' : None }),
('1ULC_co_n' ,'1ULC_co',     { 'NSW': 30 }),
('1ULC_co1' , '1ULC_co', {**mag_packet_n, 'LDIPOL': '.TRUE.'}, over), 


('1uh_co' , '1u_co',  dftu_packet_h, ''),
('1uh2_co' , '1u_co',  dftu_packet_h2, ''),
('1h_co' , '1u_co',  {**dftu_packet_off, 'NPAR' : None, 'NSIM' :None}, ''),






]










# header.varset['9ac'].printme()
#Create u-set; be cautious
def create_u_sets(based_on):
    """
    Special function for cathode projects.
    Allows to create quickly sets with different U values based on some set.
    """
    
    #Shishkin
    U_dic = {'LiFePO4':{'Fe':2.1}, 'NaFePO4':{'Fe':2.2}, 'LiMnPO4':{'Mn':2.2,}, 
             'LiCoPO4':{'Co':2.8}, 'LiTiS2':{'Ti':3.3}, 'LiNiO2':{'Ni':4.6}, 'LiCoO2':{'Co':3.6},
             'TiS2':{'Ti':3.5},  'FePO4':{'Fe':3.7}, 'CoO2':{'Co':3.9}, 'NiO2':{'Ni':4.0}, 'MnPO4':{'Mn':4.0}, 'CoPO4':{'Co':4.2}, 
    }


    U_dic2 = {#LiTiO2 #Morgan2011, LiMn2O4 Zhou2004, other azh
    # 3.4:( 'NaLiCoPO4F'),
    # (4:'Fe',3.1:'V'):('Na2FeVF7')
    }

    U_dic_12 = {#Shishkin #LiTiO2 #Morgan2011, LiMn2O4 Zhou2004, other azh
            'LiFePO4':{'Fe':2.1}, 'NaFePO4':{'Fe':2.2}, 'LiMnPO4':{'Mn':2.2,}, 
            'LiCoPO4':{'Co':2.8}, 'LiTiS2':{'Ti':3.3}, 'LiNiO2':{'Ni':4.6}, 'LiCoO2':{'Co':3.6},
            'TiS2':{'Ti':3.5},  'FePO4':{'Fe':3.7}, 'CoO2':{'Co':3.9}, 'NiO2':{'Ni':4.0}, 'MnPO4':{'Mn':4.0}, 'CoPO4':{'Co':4.2}, 
            'LiTiO2':{'Ti':4.2}, 'LiMn2O4':{'Mn':4.9}, 
            'LiVPO4F':{'V':3.1}, 'KVPO4F':{'V':3.1}, 'LiVP2O7':{'V':3.1},
            'NaMnAsO4':{'Mn':3.9},
            # 'Na2FePO4F':{'Fe':4},  'KFeSO4F':{'Fe':4},
    }



    dic =  U_dic_12
    df = pd.DataFrame(dic)
    # print list(df.columns)
    ramping_sets = []
    simple__sets = []
    for mat in df:
        u_dict = df[mat].dropna().to_dict()

        if len(u_dict.values()) > 1:
            print_and_log('Error! Please implement name conventions for multi-element LDAUU')
            raise RuntimeError
        U = u_dict.values()[0]


        ramp_set = based_on+'r'+str(U).replace('.', '-')
        simp_set = based_on+str(U).replace('.', '-')
        
        if ramp_set not in ramping_sets:
            ramping_sets.append( (ramp_set,  based_on, {'u_ramping_region':(0, U+0.00013, 0.1)})    )
            simple__sets.append( (simp_set,  based_on, {'LDAUU':u_dict })  )

    df.loc['set'] = [s[0] for s in ramping_sets] # add row with set for each material

        
    #helper

    for mat in df:
        # print mat
        if 'Li' in mat or 'Na' in mat or 'K' in mat: 
            folder = mat
            DS = mat[2:]
            if 'K' in mat:
                DS = mat[1:]
            if 'Na2' in mat:
                DS = mat[3:]
        else:
            continue

        IS_set = df[mat]['set']
        if DS in df:
            DS_set = df[DS]['set']
        else:
            DS_set = IS_set

        # print [mat, IS_set, DS, DS_set, folder],','
        # for ise in '8u', '8ue4', '8ue5', '8uL':
        # print str([mat, 'set', DS, 'set', folder]).replace("'set'", 'ise'),','




    return ramping_sets+simple__sets   
# create_u_sets('just_helper')         
#user_vasp_sets+=create_u_sets('8U')
# user_vasp_sets+=create_u_sets('8Um')
# user_vasp_sets+=create_u_sets('8Utm')






















# for phonons you should use:
# PREC = Accurate avoid wrap around errors
# LREAL = .FALSE. reciprocal space projection technique
# EDIFF = 1E-6 high accuracy required

    #May be useful
#For very accurate energy calculations:
#s.set_vaspp['ISMEAR'] = [-5, " Tetrahedron with Blochl corrections"]
#However this approach needs at least 3 kpoints and produce errors in stress tensor and forces

        #




