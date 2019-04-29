import numpy as np
import cv2 as cv
import scipy.fftpack as fftpack
import scipy.signal as signal
import glob
import os
import matplotlib.cm

import dataProc

class OpVmem(dataProc.Vmem):
    def __init__(self, path, rawSize, roiOrigin, roiSize, *args, **kwargs):
        pathList = sorted(glob.glob(os.path.join(path, '*.raww')))
        self.raw = np.zeros((len(pathList), rawSize[0]*rawSize[1]), np.uint16)
        for i, path in enumerate(pathList):
            self.raw[i, :] = np.fromfile(path, np.uint16)
        self.raw = self.raw.reshape(((len(pathList),) + rawSize + (1,)))
        self.raw = self.raw[:, roiOrigin[0]:roiOrigin[0]+roiSize[0], roiOrigin[1]:roiOrigin[1]+roiSize[1]]
        self.raw = self.raw.astype(np.float32)
        self.vmem = np.zeros_like(self.raw, np.float32)

        # map to [0, 1], different from that in opmap
        self.rawMax = np.amax(self.raw, axis=0)
        self.rawMin = np.amin(self.raw, axis=0)
        self.rawRange = (self.rawMax - self.rawMin) + (self.rawMax == self.rawMin)*1
        self.vmem = (self.rawMax - self.raw) / self.rawRange
        self.vmem = self.vmem.astype(np.float32)

        self.colorMap = None

        super(OpVmem, self).__init__(1, len(pathList), roiSize[0], roiSize[1], *args, **kwargs)

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

    def temporalFilter(self, cutFreq, kernelSize, sigma):
        # low pass Gaussian
        print('Apply spatial filter before use temporal filter!')
        if not sigma > 0:
            sigma = 0.3*((kernelSize-1)*0.5 - 1) + 0.8 # same as opencv
        gaussian_kernel = signal.gaussian(kernelSize, sigma)
        kernel = np.zeros(self.length, np.float32)
        halfSize = kernelSize // 2
        kernel[0:cutFreq-halfSize] = 1
        kernel[-cutFreq+halfSize: ] = 1
        kernel[cutFreq-halfSize:cutFreq+1] = gaussian_kernel[halfSize:]
        kernel[self.length-cutFreq-1: self.length-cutFreq+halfSize] = gaussian_kernel[0:halfSize+1]
        self.kernel = kernel[:, np.newaxis, np.newaxis, np.newaxis]
        f = fftpack.fft(self.vmem, axis=0)
        f_filtered = np.multiply(f, self.kernel)
        self.vmem = np.abs(fftpack.ifft(f_filtered, axis=0), dtype=np.float32)
    
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