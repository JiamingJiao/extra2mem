import numpy as np
import os
import glob
import sys

import pseudoEcg
import dataProc
sys.path.append('../')
import dataProc

def getSimInSubArea(sim_path_list, dst_path, size):
    if not os.path.exists(os.path.join(dst_path, 'phie')):
        os.makedirs(os.path.join(dst_path, 'phie'))
    if not os.path.exists(os.path.join(dst_path, 'vmem')):
        os.makedirs(os.path.join(dst_path, 'vmem'))
    for sim_path, k in enumerate(sim_path_list):
        src_phie_path = os.path.join(sim_path, 'phie_')
        phie = dataProc.loadData(src_phie_path)
        src_vmem_path = os.path.join(sim_path, 'vmem_')
        vmem = dataProc.loadData(src_vmem_path)
        for i in range(0, phie.shape[1], size[0]): # rows
            for j in range(0, phie.shape[1], size[1]): #columns
                dst_phie_path = os.path.join(dst_path, 'phie', '%02d_%03d_%03d'%(k, i, j))
                np.save(dst_phie_path, phie[:, i:i+size[0], j:j+size[1], :])
                dst_vmem_path = os.path.join(dst_path, 'vmem', '%02d_%03d_%03d'%(k, i, j))
                np.save(dst_vmem_path, vmem[:, i:i+size[0], j:j+size[1], :])

def getPecg(src_path, dst_path, elec_pos, gnd_pos, conductance):
    src_path_list = sorted(glob.glob(os.path.join(src_path, '*.npy')))
    for src_path in src_path_list:
        phie = np.load(src_path)
        pecg_no_ref = pseudoEcg.calcPecgSequence(phie, elec_pos, conductance)
        pecg_ref = pseudoEcg.calcPecgSequence(phie, gnd_pos, conductance)
        pecg = np.subtract(pecg_no_ref, pecg_ref)
        np.save(os.path.join(dst_path, src_path[-14:-5]), pecg)