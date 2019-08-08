import numpy as np
import scipy
import scipy.signal as signal
import sys
sys.path.append('./preprocess/')
import preprocess.pseudoEcg as pseudoEcg
import dataProc


def findPeaks(src, **kwargs):
    # src shape: time*channels
    peaks = []
    for i in range(0, src.shape[1]):
        peaks_channel, _ = signal.find_peaks(src[:, i], **kwargs)
        peaks.append(peaks_channel)
    return peaks


def activationToVmem(src, vmem_min, vmem_max, phase0_duration, length):
    channels_num = len(src)
    vmem_full = np.zeros((length, channels_num), np.float32)
    time_min = []
    time_max = []
    for i, peaks_channel in enumerate(src):
        time_idx = []
        for peak in peaks_channel:
            time_idx.append(peak - phase0_duration)
            time_idx.append(peak)
        vmem_peaks = list((vmem_min, vmem_max, ) * len(peaks_channel))
        if time_idx[0]<1:
            del time_idx[0]
            del vmem_peaks[0]
        interpolate = scipy.interpolate.interp1d(time_idx, vmem_peaks, 'linear', bounds_error=False)
        vmem_full[time_idx[0]:time_idx[-1], i] = interpolate(np.linspace(time_idx[0], time_idx[-1], time_idx[-1]-time_idx[0], False))
        time_min.append(time_idx[0])
        time_max.append(time_idx[-1])
    start = max(time_min)
    end = min(time_max)  # start>end is possible
    # mask = np.zeros_like(vmem_full, np.bool)
    # mask[0:start, ...] = True
    # mask[end+1:, ...] = True
    # vmem_valid = np.ma.masked_where(mask, vmem_full)
    return vmem_full, (start, end)


class VmemFromActivation(object):
    def __init__(self, ecg, electrodes_pos, size, vmem_min=-80, vmem_max=20, phase0_duration=2, interpolation_method='cubic', **find_peaks_kwargs):
        self.ecg = ecg
        self.ecg_derivative = dataProc.channelNormalize(np.gradient(ecg, axis=0))
        self.activation_time = findPeaks(self.ecg_derivative, **find_peaks_kwargs)
        vmem_electrodes, time_range = activationToVmem(self.activation_time, vmem_min, vmem_max, phase0_duration, self.ecg.shape[0])
        self.start = time_range[0]
        self.end = time_range[1]+1
        self.vmem = pseudoEcg.interpolate(vmem_electrodes, electrodes_pos, size, interpolation_method)
        self.mask = np.where(np.abs(self.vmem-self.vmem)<np.finfo(self.vmem.dtype).eps, False, True)
        self.vmem = np.ma.masked_where(self.mask, self.vmem)
