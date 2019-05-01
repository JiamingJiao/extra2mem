import numpy as np
import os
import glob
import sys

import dataProc
sys.path.append('./preprocess/')
import pseudoEcg

def getSimInSubArea(sim_path_list, dst_path, size):
    if not os.path.exists(os.path.join(dst_path, 'phie')):
        os.makedirs(os.path.join(dst_path, 'phie'))
    if not os.path.exists(os.path.join(dst_path, 'vmem')):
        os.makedirs(os.path.join(dst_path, 'vmem'))
    for k, sim_path in enumerate(sim_path_list):
        src_phie_path = os.path.join(sim_path, 'phie_')
        phie = dataProc.loadData(src_phie_path)
        src_vmem_path = os.path.join(sim_path, 'vmem_')
        vmem = dataProc.loadData(src_vmem_path)
        for i in range(0, phie.shape[1]-size[0], size[0]): # rows
            for j in range(0, phie.shape[1]-size[1], size[1]): #columns
                dst_phie_path = os.path.join(dst_path, 'phie', '%02d_%03d_%03d'%(k, i, j))
                np.save(dst_phie_path, phie[:, i:i+size[0], j:j+size[1], :])
                dst_vmem_path = os.path.join(dst_path, 'vmem', '%02d_%03d_%03d'%(k, i, j))
                np.save(dst_vmem_path, vmem[:, i:i+size[0], j:j+size[1], :])

def getPecg(src_path, dst_path, elec_pos, gnd_pos, conductance, inter_size):
    src_path_list = sorted(glob.glob(os.path.join(src_path, '*.npy')))
    dst_pecg_folder = os.path.join(dst_path, 'pecg')
    if not os.path.exists(dst_pecg_folder):
        os.makedirs(dst_pecg_folder)
    for src_path in src_path_list:
        phie = np.load(src_path)
        pecg_no_ref = pseudoEcg.calcPecgSequence(phie, elec_pos, conductance)
        pecg_ref = pseudoEcg.calcPecgSequence(phie, gnd_pos, conductance)
        pecg = np.subtract(pecg_no_ref, pecg_ref)
        pecg_map = pseudoEcg.interpolate(pecg, elec_pos[:, 0:2], inter_size)
        np.save(os.path.join(dst_pecg_folder, src_path[-14:-4]), pecg_map)

def get3dBlocks(src_path, length, return_data=True, save=False, dst_path=None):
    file_names = sorted(glob.glob(os.path.join(src_path, '*.npy')))
    array_list = []
    blocks_num = 0
    for file_name in file_names:
        temp = np.load(file_name)
        array_list.append(temp)
        blocks_num += temp.shape[0] // length
    dst = np.zeros(((blocks_num, length,)+temp.shape[1:4]), dataProc.DATA_TYPE)
    block_cnt = 0
    for data in array_list:
        for k in range(0, data.shape[0]-length, length):
            dst[block_cnt, :, :, :, :] = data[k:k+length, :, :, :]
            block_cnt += 1
    if save:
        for k, block in enumerate(dst):
            np.save(os.path.join(dst_path, '%06d'%k), block)
    if return_data:
        return dst