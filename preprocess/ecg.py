import pandas
import numpy as np
import scipy.signal as signal
import sys

sys.path.append('./preprocess/')
import pseudoEcg

def getEcgMaps(src_path, elec_pos, size):
    data = load(src_path)
    dst = pseudoEcg.interpolate(data, elec_pos, size)
    return dst

def filterEcg(src, sampling_rate, fc_low, fc_notch_list, sigma_list):
    filter_low = makeLowpassFilter(sampling_rate, src.shape[0], fc_low, order=5)
    filtered_low = signal.filtfilt(*filter_low, src, axis=0)
    filter_notch = makeNotchFilter()

def load(path):
    csv_data = pandas.read_csv(path, skiprows=12, header=None)
    trigger = np.array(csv_data.iloc[0:-1, 2])
    start = np.argmax(trigger<-5) + 10
    dst = -np.array(csv_data.iloc[start:start+2000, 3:-1])
    return dst

def makeLowpassFilter(sampling_rate, length, f_cut, order):
    time = np.arange(length) / sampling_rate
    w = f_cut / (sampling_rate/2)
    b, a = signal.butter(order, w, 'lowpass')
    return b, a

def makeNotchFilter(pos_list, sigma_list, length):
    # Gaussian notch filter, apply on unshifted fft result
    assert len(pos_list, sigma_list), 'length of pos != length of sigma'
    kernel = np.ones((length))
    ksize = int((((max(sigma_list)-0.8)/0.3 + 1)**2 + 2)/2)*2 + 1 # same as OpenCV
    for pos, sigma in zip(pos_list, sigma_list):
        gaussian_kernel = 1-signal.gaussian(ksize, sigma)
        one_notch = kernel[pos-ksize//2:pos+ksize//2+1]
        kernel[pos-ksize//2:pos+ksize//2+1] = np.minimum(one_notch, gaussian_kernel)
    dst = np.concatenate((kernel, np.flip(kernel)))[:, np.newaxis]
    return dst
