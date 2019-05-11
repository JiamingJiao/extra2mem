import numpy as np
import cv2 as cv

def rotate(src, center, angle, inter_flag, dst):
    # must pad src!!!
    # pos is the position of left top point
    trans_mat = cv.getRotationMatrix2D(center, angle, 1)
    #d_size = src.shape
    #dst = np.zeros((d_size), np.float32)
    cv.warpAffine(src, trans_mat, dst.shape[0:2], dst, inter_flag)
    #return dst