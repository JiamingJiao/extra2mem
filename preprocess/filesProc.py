import numpy as np
import os
import glob
import sys
import cv2 as cv
import dataProc
sys.path.append('./preprocess/')
import pseudoEcg
import rotate


def getRotatedSimInSubArea(sim_path_list, dst_path, size, angle, src_size=200, inter_flag=cv.INTER_LINEAR):
    assert angle>-360 and angle<360, 'angle must be between -360 and 360'
    if angle<0:
        angle_p = angle + 360
    else:
        angle_p = angle
    angle_acute = angle_p % 90

    padding_size = int(0.25*src_size)
    padding_array = ((0, 0), (padding_size,)*2, (padding_size,)*2, (0, 0))
    contour = rotate.getContour(padding_size, padding_size+src_size)
    contour_rotated = rotate.rotateContour(contour, angle, 2*padding_size+src_size)
    top_contour_i, top_contour_j = contour_rotated[angle_p//90, -1, :, 0]
    rad = angle_acute*np.pi/180
    inscribed_square_size = src_size / (np.sin(rad) + np.cos(rad))
    i_start = int(top_contour_i + inscribed_square_size*np.cos(rad)*np.sin(rad))
    j_start = int(top_contour_j - inscribed_square_size*np.cos(rad)**2)
    i_end = int(i_start + inscribed_square_size)
    j_end = int(j_start + inscribed_square_size)

    if not os.path.exists(os.path.join(dst_path, 'phie')):
        os.makedirs(os.path.join(dst_path, 'phie'))
    if not os.path.exists(os.path.join(dst_path, 'vmem')):
        os.makedirs(os.path.join(dst_path, 'vmem'))
    for k, sim_path in enumerate(sim_path_list):
        src_phie_path = os.path.join(sim_path, 'phie_')
        phie = dataProc.loadData(src_phie_path)
        phie_padded = np.pad(phie, padding_array, 'constant')

        src_vmem_path = os.path.join(sim_path, 'vmem_')
        vmem = dataProc.loadData(src_vmem_path)
        vmem_padded = np.pad(vmem, padding_array, 'constant')

        phie_rotated = np.zeros_like(phie_padded)
        vmem_rotated = np.zeros_like(vmem_padded)
        trans_mat = cv.getRotationMatrix2D((phie_rotated.shape[1]//2,)*2, angle, 1)
        for (phie_frame, phie_r_frame, vmem_frame, vmem_r_frame) in zip(phie_padded, phie_rotated, vmem_padded, vmem_rotated):
            cv.warpAffine(phie_frame, trans_mat, phie_frame.shape[0:2], phie_r_frame, inter_flag)
            cv.warpAffine(vmem_frame, trans_mat, vmem_frame.shape[0:2], vmem_r_frame, inter_flag)

        for i in range(i_start, i_end-size, size):  # rows
            for j in range(j_start, j_end-size, size):  # columns
                dst_phie_path = os.path.join(dst_path, 'phie', '%02d_%02d_%03d_%03d'%(angle, k, i, j))
                np.save(dst_phie_path, phie_rotated[:, i:i+size, j:j+size, :])
                dst_vmem_path = os.path.join(dst_path, 'vmem', '%02d_%02d_%03d_%03d'%(angle, k, i, j))
                np.save(dst_vmem_path, vmem_rotated[:, i:i+size, j:j+size, :])


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
        np.save(os.path.join(dst_pecg_folder, src_path.split('/')[-1][:-4]), pecg_map)


def getBinaryPecg(src_path, dst_path, elec_pos, gnd_pos, conductance, inter_size, **find_peaks_args):
    src_path_list = sorted(glob.glob(os.path.join(src_path, '*.npy')))
    dst_pecg_folder = os.path.join(dst_path, 'pecg_bin')
    if not os.path.exists(dst_pecg_folder):
        os.makedirs(dst_pecg_folder)
    for src_path in src_path_list:
        phie = np.load(src_path)
        pecg_no_ref = pseudoEcg.calcPecgSequence(phie, elec_pos, conductance)
        pecg_ref = pseudoEcg.calcPecgSequence(phie, gnd_pos, conductance)
        pecg = np.subtract(pecg_no_ref, pecg_ref)
        pecg_binary = pseudoEcg.binarize(pecg, **find_peaks_args)
        pecg_map = pseudoEcg.interpolate(pecg_binary, elec_pos[:, 0:2], inter_size)
        np.save(os.path.join(dst_pecg_folder, src_path.split('/')[-1][:-4]), pecg_map)


def get3dBlocks(src_path, length):
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
    return dst


def save3dBlocks(src_path, length, dst_path, norm_range, resize=False, dsize=None):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    file_names = sorted(glob.glob(os.path.join(src_path, '*.npy')))
    blocks_cnt = 0
    for file_name in file_names:
        src = np.load(file_name)
        src = (src-norm_range[0]) / (norm_range[1]-norm_range[0])
        if resize:
            resized = np.zeros((src.shape[0], dsize[0], dsize[1], src.shape[3]))
            for k in range(0, src.shape[0]):
                cv.resize(src[k], dsize, resized[k], 0, 0, cv.INTER_LINEAR)
        else:
            resized = src
        for k in range(0, resized.shape[0]-length, length):
            np.save(os.path.join(dst_path, '%06d'%blocks_cnt), resized[k:k+length])
            blocks_cnt += 1
