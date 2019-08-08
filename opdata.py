import numpy as np
import cv2 as cv
# import scipy.fftpack as fftpack
import scipy
import scipy.signal as signal
import glob
import os
import matplotlib.cm


class OpVmem(object):
    def __init__(self, path, rawSize, roi, sampling_rate=1000, start=0, end=0, resize=False, dsize=(256, 256), *args, **kwargs):
        pathList = sorted(glob.glob(os.path.join(path, '*.raww')))
        if not end>0:
            end = len(pathList)
        pathList = pathList[start:end]
        self.raw = np.zeros((len(pathList), rawSize[0]*rawSize[1]), np.uint16)
        self.sampling_rate = sampling_rate
        for i, path in enumerate(pathList):
            self.raw[i, :] = np.fromfile(path, np.uint16)
        self.raw = self.raw.reshape(((len(pathList),) + rawSize + (1,)))
        self.raw = self.raw[:, roi[0]:roi[2], roi[1]:roi[3]]
        self.raw = self.raw.astype(np.float32)

        # map to [0, 1], different from that in opmap
        self.rawMax = np.amax(self.raw, axis=0)
        self.rawMin = np.amin(self.raw, axis=0)
        self.rawRange = (self.rawMax - self.rawMin) + (self.rawMax == self.rawMin)*1
        vmem_raw_size = (self.rawMax - self.raw) / self.rawRange
        vmem_raw_size = vmem_raw_size.astype(np.float32)
        self.vmem = np.zeros((self.raw.shape[0], dsize[0], dsize[1], 1), np.float32)
        if resize:
            for k, raw_frame in enumerate(vmem_raw_size):
                cv.resize(raw_frame, dsize, self.vmem[k], interpolation=cv.INTER_CUBIC)
        else:
            self.vmem = vmem_raw_size

        self.mask = np.ones((self.vmem.shape[1:3]), np.uint8)
        self.colorMap = None
        self.kernel = None

    def setRoi(self, threshold=0.1, dilation_iterations=3, erosion_iterations=20):
        thres = threshold*np.max(self.rawRange)
        mask = np.where(self.rawRange[..., 0]>thres, 1, 0).astype(np.uint8)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        cv.dilate(mask, kernel, mask, iterations=dilation_iterations)
        cv.erode(mask, kernel, mask, iterations=erosion_iterations)
        self.mask = cv.resize(mask, self.vmem.shape[1:3], interpolation=cv.INTER_NEAREST)
        np.multiply(self.vmem, self.mask[..., np.newaxis], self.vmem)

    def spatialFilter(self, sigma):
        kernel_size = int(sigma*3 + 0.5)
        for frame in self.vmem:
            cv.GaussianBlur(frame, (kernel_size, kernel_size), sigma, frame, sigma, cv.BORDER_REPLICATE)

    def temporalFilter(self, sigma):
        self.vmem = scipy.ndimage.gaussian_filter1d(self.vmem, sigma, 0, mode='nearest', truncate=4)

    def highPassFilter(self, f_cut, order):
        w = f_cut / (self.sampling_rate/2)
        b, a = signal.butter(order, w, 'highpass')
        self.vmem = signal.filtfilt(b, a, self.vmem, axis=0)

    def setColor(self, cmap='inferno'):
        mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
        self.colorMap = np.empty((self.length, self.height, self.width, 4), np.float32)
        for i, frame in enumerate(self.vmem[:, :, :, 0]):
            self.colorMap[i] = mapper.to_rgba(frame, norm=False)
