import numpy as np
import cv2 as cv

PECG_CONV_KERNEL = np.zeros((3, 3, 1), np.float32)
PECG_CONV_KERNEL[1, :, 0] = 1
PECG_CONV_KERNEL[:, 1, 0] = 1
PECG_CONV_KERNEL[1, 1, 0] = -4

def calcPecgSequence(phie, pos_array, conductance):
	points_num = pos_array.shape[0]
	dst = np.zeros((phie.shape[0], points_num), np.float32)
	distance = calcDistance(pos_array, phie.shape[1:])
	for i, phie_frame in enumerate(phie):
		dst[i] = calcPecgFrame(phie_frame, distance, conductance)
	return dst

def calcPecgFrame(phie, distance, conductance):
	points_num = distance.shape[0]
	map_shape = phie.shape
	diff_v = np.ndarray(map_shape, np.float32)
	cv.filter2D(phie, cv.CV_32FC1, PECG_CONV_KERNEL, diff_v, (-1, -1), 0, cv.BORDER_REPLICATE)
	quotient = np.zeros(map_shape, np.float32)
	dst = np.zeros((points_num), np.float32)
	for i in range(0, points_num):
		np.divide(diff_v, distance[i], quotient)
		dst[i] = np.sum(quotient)
	# np.multiply(dst, conductance, dst)
	return dst

def calcDistance(pos_array, map_shape):
	map_xy = np.array(makeGrid(map_shape), np.float32)
	map_xy = np.swapaxes(map_xy, 0, 2)
	map_xyz = np.zeros((map_shape[0:2]+(3,)), np.float32)
	map_xyz[:, :, 0:2] = map_xy
	map_xyz[:, :, 2] = 0
	dst = np.zeros(((pos_array.shape[0],)+map_shape), dtype=np.float32)
	for i, pos in enumerate(pos_array):
		dst[i, :, :, 0] = np.linalg.norm(pos-map_xyz, axis=2)
	return dst

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