import numpy as np
import cv2 as cv
import scipy.fftpack as fftpack
import scipy.signal as signal
import glob
import os
import matplotlib.cm
import sys

import dataProc
sys.path.append('./preprocess/')
import ecg

class OpVmem(dataProc.Vmem):
    def __init__(self, path, rawSize, roi, sampling_rate=1000, *args, **kwargs):
        pathList = sorted(glob.glob(os.path.join(path, '*.raww')))
        self.raw = np.zeros((len(pathList), rawSize[0]*rawSize[1]), np.uint16)
        self.sampling_rate = sampling_rate
        for i, path in enumerate(pathList):
            self.raw[i, :] = np.fromfile(path, np.uint16)
        self.raw = self.raw.reshape(((len(pathList),) + rawSize + (1,)))
        self.raw = self.raw[:, roi[0]:roi[2], roi[1]:roi[3]]
        self.raw = self.raw.astype(np.float32)
        self.vmem = np.zeros_like(self.raw, np.float32)

        # map to [0, 1], different from that in opmap
        self.rawMax = np.amax(self.raw, axis=0)
        self.rawMin = np.amin(self.raw, axis=0)
        self.rawRange = (self.rawMax - self.rawMin) + (self.rawMax == self.rawMin)*1
        self.vmem = (self.rawMax - self.raw) / self.rawRange
        self.vmem = self.vmem.astype(np.float32)

        self.colorMap = None

        super(OpVmem, self).__init__(1, len(pathList), roi[2]-roi[0], roi[3]-roi[1], *args, **kwargs)

        self.kernel = None
    
    def spatialFilter(self, kernelSize, sigma):
        
        if not sigma > 0:
            sigma = 0.3*((kernelSize-1)*0.5 - 1) + 0.8 # same as opencv
        for frame in self.vmem:
            cv.GaussianBlur(frame, (kernelSize, kernelSize), sigma, frame, sigma, cv.BORDER_REPLICATE)
        '''
        for frame in self.vmem:
            cv.medianBlur(frame, kernelSize, frame)
        '''

    # def temporalFilter(self, fcl, fch, order):
    #     # # lowpass butterworth
    #     # b, a = ecg.makeLowpassFilter(self.sampling_rate, self.length, fcl, order_l)
    #     # self.vmem = signal.filtfilt(b, a, self.vmem, 0)
    #     # # highpass butterworth
    #     # wc = fch / (self.sampling_rate/2)
    #     # b, a = signal.butter(order_h, w, 'highpass')
    #     # self.vmem = signal.filtfilt(b, a, self.vmem, 0)
    #     wcl = fcl / (self.sampling_rate/2)
    #     wch = fch / (self.sampling_rate/2)
    #     b, a = signal.butter(order, (wcl, wch), 'bandpass')
    #     self.vmem = signal.filtfilt(b, a, self.vmem, 0)

    def temporalFilter(self, kernel_size, sigma):
        assert kernel_size%2==1, 'kernel size must be a odd number' 
        padding_size = kernel_size//2
        padded = np.pad(self.vmem, ((padding_size, padding_size), (0, 0), (0, 0), (0, 0)), 'edge')
        kernel = np.ones((kernel_size, 1, 1, 1), dtype=np.float32) / kernel_size
        # if not sigma > 0:
        #     sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8 # same as opencv
        # kernel = signal.gaussian(kernel_size, sigma)[:, np.newaxis, np.newaxis, np.newaxis]
        self.vmem = signal.convolve(padded, kernel, 'valid')
    
    def setColor(self, cmap='inferno'):
        mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
        self.colorMap = np.empty((self.length, self.height, self.width, 4), np.float32)
        for i, frame in enumerate(self.vmem[:, :, :, 0]):
            self.colorMap[i] = mapper.to_rgba(frame, norm=False)

def findPeak(src, pos_array):
    dst = []
    for pos in pos_array:
        peaks, _ = signal.find_peaks(src[:, pos[0], pos[1], 0], width=(30, None), height=(0.05, None), distance=100, prominence=(0.1, None))
        dst.append(peaks)
    return dst