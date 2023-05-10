"""
Convert PDB file to HDF5 file.
File path 1 is the PDB folder, and file path 2 is the folder where the HDF5 file to be generated is located.
"""
import sys
import os
import h5py
import numpy as np
import copy
import warnings
# from mdtraj import mdconvert
from PyProtein import *
from PyPeriodicTable import *

# path1 = '/dssg/home/acct-clschf/clschf/zy/AlphafoldData/newData/'
# path2 = '/dssg/home/acct-clschf/clschf/zy/AlphafoldData/H5pydata/'
path1 = 'F:/AlphaFoldData/CATHTAR/CATH_30835/'
path2 = 'F:/AlphaFoldData/CATHTAR/HDF5/'

dirs = os.listdir(path1)
periodicTable = PyPeriodicTable()
for dir in dirs:
    if '.pdb' in dir:
        filename = dir.replace(".pdb", "")
        curProtein = PyProtein(periodicTable)
        curFile = path1+dir
        hdf5File = path2+filename+".hdf5"
        curProtein.load_molecular_file(curFile, pLoadAnim=True, pFileType="pdb", pLoadHydrogens=False, pLoadH2O=False, pBackBoneOnly=False, pChainFilter=None)
        curProtein.compute_covalent_bonds()
        curProtein.compute_hydrogen_bonds()
        curProtein.save_hdf5(hdf5File)

#         filename = dir.replace(".pdb", "")
#         with open(path1+dir, 'r') as pdb_file:
#             pdb_data = pdb_file.readlines()
#
#         with h5py.File(path2+filename+".hdf5", 'w') as hdf5_file:
#             hdf5_file.create_dataset('pdb_data', data=pdb_data)
#
#         pdb_file.close()
#         hdf5_file.close()

