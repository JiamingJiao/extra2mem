import numpy as np
import cv2 as cv


def rotateMap(src, angle, inter_flag, dst):
    # rotate src around its center point
    # must pad src!!!
    trans_mat = cv.getRotationMatrix2D((src.shape[0]//2, src.shape[1]//2), angle, 1)
    cv.warpAffine(src, trans_mat, dst.shape[0:2], dst, inter_flag)


def rotateContour(src, angle, map_size):
    # counterclockwise
    # src.ndim >= 3
    rad = angle*np.pi/180
    cos_angle = np.cos(rad)
    sin_angle = np.sin(rad)
    rotation_vec = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]], np.float32)
    rotated_src = np.matmul(rotation_vec, src.astype(np.float32))
    src_center = np.array([[map_size/2], [map_size/2]], np.float32)
    center_displacement = np.matmul(rotation_vec, src_center) - src_center
    dst = (rotated_src - center_displacement).astype(np.uint16)
    return dst


def getContour(start, end):
    # Assume that height = width, i_start = j_start
    height = end - start + 1
    dst = np.ndarray((4, height, 2, 1), np.uint16)
    # top
    dst[0, :, 0, 0] = start
    dst[0, :, 1, 0] = np.linspace(start, end, height, dtype=np.uint16)
    # right
    dst[1, :, 0, 0] = dst[0, :, 1, 0]
    dst[1, :, 1, 0] = end
    # bottom
    dst[2, :, 0, 0] = end
    dst[2, :, 1, 0] = np.linspace(end, start, height, dtype=np.uint16)
    # left
    dst[3, :, 0, 0] = dst[2, :, 1, 0]
    dst[3, :, 1, 0] = start
    # dst = dst.reshape((4*height, 2, 1))
    return dst
