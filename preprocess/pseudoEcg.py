import numpy as np
# import cv2 as cv
import scipy.interpolate
import scipy.signal as signal

import dataProc


# # --------------------------------------------
# PECG_CONV_KERNEL = np.zeros((3, 3, 1), np.float32)
# PECG_CONV_KERNEL[1, :, 0] = 1
# PECG_CONV_KERNEL[:, 1, 0] = 1
# PECG_CONV_KERNEL[1, 1, 0] = -4


# def calcPecgSequence(phie, pos_array, conductance):
#     points_num = pos_array.shape[0]
#     dst = np.zeros((phie.shape[0], points_num), np.float32)
#     distance = calcDistance(pos_array, phie.shape[1:])
#     for i, phie_frame in enumerate(phie):
#         dst[i] = calcPecgFrame(phie_frame, distance, conductance)
#     return dst


# def calcPecgFrame(phie, distance, conductance):
#     points_num = distance.shape[0]
#     map_shape = phie.shape
#     diff_v = np.ndarray(map_shape, np.float32)
#     cv.filter2D(phie, cv.CV_32FC1, PECG_CONV_KERNEL, diff_v, (-1, -1), 0, cv.BORDER_REPLICATE)
#     quotient = np.zeros(map_shape, np.float32)
#     dst = np.zeros((points_num), np.float32)
#     for i in range(0, points_num):
#         np.divide(diff_v, distance[i], quotient)
#         dst[i] = np.sum(quotient)
#     # np.multiply(dst, conductance, dst)
#     return dst


# def calcDistance(pos_array, map_shape):
#     map_xy = np.array(makeGrid(map_shape), np.float32)
#     map_xy = np.swapaxes(map_xy, 0, 2)
#     map_xyz = np.zeros((map_shape[0:2]+(3,)), np.float32)
#     map_xyz[:, :, 0:2] = map_xy
#     map_xyz[:, :, 2] = 0
#     dst = np.zeros(((pos_array.shape[0],)+map_shape), dtype=np.float32)
#     for i, pos in enumerate(pos_array):
#         dst[i, :, :, 0] = np.linalg.norm(pos-map_xyz, axis=2)
#     return dst
# # --------------------------------------------


# --------------------------------------------
def calcPecgSequence(phie, pos_array):
    points_num = pos_array.shape[0]
    dst = np.zeros((phie.shape[0], points_num), np.float32)
    r_arrs = calcDistanceVectorArrays(pos_array, phie.shape[1:3])
    for k, phie_frame in enumerate(phie[..., 0]):
        dst[k] = calcPecgFrame(phie_frame, r_arrs)
    return dst


def calcPecgFrame(phie, r_arrs):
    phie_grad_x, phie_grad_y = np.gradient(phie, axis=(0, 1))
    phie_grad_z = np.zeros_like(phie_grad_x)
    dst = np.zeros((r_arrs.shape[0]), np.float32)
    for i, r_arr in enumerate(r_arrs):
        elements = phie_grad_x*r_arr[..., 0] + phie_grad_y*r_arr[..., 1] + phie_grad_z*r_arr[..., 2]
        dst[i] = np.sum(elements)
    dst = -1*dst
    return dst


def calcDistanceVectorArrays(pos_array, map_shape):
    map_xy = np.array(makeGrid(map_shape), np.float32)
    map_xy = np.swapaxes(map_xy, 0, 2)
    map_xyz = np.zeros((map_shape[0:2]+(3,)), np.float32)
    map_xyz[:, :, 0:2] = map_xy
    map_xyz[:, :, 2] = 0
    r_arrs = np.zeros((pos_array.shape[0], map_shape[0], map_shape[1], 3), dtype=np.float32)
    for i, pos in enumerate(pos_array):
        r_arrs[i] = pos[np.newaxis, np.newaxis, :] - map_xyz
    r_arrs_norm_cubic = np.linalg.norm(r_arrs, 2, -1, True) ** 3
    dst = r_arrs / r_arrs_norm_cubic
    return dst
# ------------------------------------------


def makePosArray(electrodes_pos, z):
    dst = np.zeros((electrodes_pos.shape[0], 3), np.uint16)
    dst[:, 0:2] = electrodes_pos
    dst[:, 2] = z
    return dst


def makeGrid(shape):
    x = np.linspace(0, shape[0], shape[0], False, dtype=np.uint16)
    y = np.linspace(0, shape[1], shape[1], False, dtype=np.uint16)
    dst = np.meshgrid(x, y)
    return dst


def addZ(xy_array, z):
    dst = np.zeros((xy_array.shape[0], 3), np.uint16)
    dst[:, 0:2] = xy_array
    dst[:, 2] = z
    return dst


def interpolate(data, pos_array, size, interpolation_method='nearest', **interpolation_args):
    first_row_idx = np.linspace(0, size[0], num=size[0], endpoint=False, dtype=dataProc.DATA_TYPE)
    first_col_idx = np.linspace(0, size[1], num=size[1], endpoint=False, dtype=dataProc.DATA_TYPE)
    col_idx, row_idx = np.meshgrid(first_row_idx, first_col_idx)
    grid = (row_idx, col_idx)
    dst = np.zeros((data.shape[0], size[0], size[1], 1), data.dtype)
    for k, frame in enumerate(data):
        dst[k, :, :, 0] = scipy.interpolate.griddata(pos_array, frame, grid, method=interpolation_method, **interpolation_args)
    return dst


def binarize(src, **find_peaks_args):
    dst = np.zeros_like(src, src.dtype)
    dst[:, :] = 0.5
    for k in range(src.shape[-1]):
        peaks, _ = signal.find_peaks(src[..., k], **find_peaks_args)
        dst[peaks, k] = 1
        valleys, _ = signal.find_peaks(-src[..., k], **find_peaks_args)
        dst[valleys, k] = 0
    return dst
