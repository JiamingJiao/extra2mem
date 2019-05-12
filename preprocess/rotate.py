import numpy as np
import cv2 as cv
import math

def rotateMap(src, angle, inter_flag, dst):
    # must pad src!!!
    trans_mat = cv.getRotationMatrix2D((src.shape[0]//2, src.shape[1]//2), angle, 1)
    cv.warpAffine(src, trans_mat, dst.shape[0:2], dst, inter_flag)

def rotateContour(src, angle, dst):
    # src.ndim >= 3
    #assert src.shape == dst.shape, 'src and dst have different shapes!'
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rotation_vec = np.array([[cos_angle, sin_angle], [cos_angle, -sin_angle]], np.float32)
    np.matmul(rotation_vec, src, dst)

def getContour(start, end):
    # Assume that height = width, i_start = j_start
    height = end - start + 1
    dst = np.ndarray((4, height, 2, 1), np.uint16)
    # left
    dst[0, :, 0, 0] = np.linspace(start, end, height, dtype=np.uint16)
    dst[0, :, 1, 0] = start
    # bottom
    dst[1, :, 0, 0] = end
    dst[1, :, 1, 0] = dst[0, :, 0, 0]
    # right
    dst[2, :, 0, 0] = np.linspace(end, start, height, dtype=np.uint16)
    dst[2, :, 1, 0] = end
    # top
    dst[3, :, 0, 0] = start
    dst[3, :, 1, 0] = dst[2, :, 0, 0]