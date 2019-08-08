import pandas
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import glob
import os
import sys

sys.path.append('./preprocess/')
import pseudoEcg


def getEcgMaps(src_path, start, end, elec_pos, size, **read_scv_kwargs):
    data = load(src_path, start, end, **read_scv_kwargs)
    dst = pseudoEcg.interpolate(data, elec_pos, size)
    return dst


def filterEcg(src, sampling_rate, fc_low, order_low, fc_high, order_high, fc_notch_list, sigma_list):
    # if fc_low==-1 or order_low==-1:
    #     filtered_low = src
    # else:
    #     filter_low = makeLowpassFilter(sampling_rate, src.shape[0], fc_low, order_low)
    #     filtered_low = signal.filtfilt(*filter_low, src, axis=0)

    filter_low = makeLowpassFilter(sampling_rate, src.shape[0], fc_low, order_low)
    filtered_low = signal.filtfilt(*filter_low, src, axis=0)
    filter_high = makeHighpassFilter(sampling_rate, src.shape[0], fc_high, order_high)
    filtered_high = signal.filtfilt(*filter_high, filtered_low, axis=0)
    notch_pos_list = [int(fc_notch*src.shape[0]/sampling_rate) for fc_notch in fc_notch_list]
    filter_notch = makeNotchFilter(notch_pos_list, sigma_list, src.shape[0])
    filtered_high_f = fftpack.fft(filtered_high, axis=0)
    filtered_f = np.multiply(filtered_high_f, filter_notch)
    dst = fftpack.ifft(filtered_f, axis=0).real.astype(np.float32)
    return dst


def load(path, start, end, **read_csv_kwargs):
    file_name = glob.glob(os.path.join(path, '*csv'))[0]
    csv_data = pandas.read_csv(file_name, skiprows=12, header=None, **read_csv_kwargs)
    trigger = np.array(csv_data.iloc[0:-1, 2])
    triggered_idx = np.argmax(trigger<-5)
    dst = -np.array(csv_data.iloc[triggered_idx+start:triggered_idx+end, 3:-1])
    return dst  # (time, channel)


def makeLowpassFilter(sampling_rate, length, f_cut, order):
    w = f_cut / (sampling_rate/2)
    b, a = signal.butter(order, w, 'lowpass')
    return b, a


def makeHighpassFilter(sampling_rate, length, f_cut, order):
    w = f_cut / (sampling_rate/2)
    b, a = signal.butter(order, w, 'highpass')
    return b, a


def makeNotchFilter(pos_list, sigma_list, length):
    # Gaussian notch filter, apply on unshifted fft result
    assert len(pos_list) == len(sigma_list), 'length of pos != length of sigma'
    kernel = np.ones((length//2))
    ksize = int((((max(sigma_list)-0.8)/0.3 + 1)**2 + 2)/2)*2 + 1  # same as OpenCV
    for pos, sigma in zip(pos_list, sigma_list):
        gaussian_kernel = 1-signal.gaussian(ksize, sigma)
        one_notch = kernel[pos-ksize//2:pos+ksize//2+1]
        kernel[pos-ksize//2:pos+ksize//2+1] = np.minimum(one_notch, gaussian_kernel)
    dst = np.concatenate((kernel, np.flip(kernel, 0)))[:, np.newaxis]
    return dst


def notchFilter(src, sampling_rate, notch_list, sigma_list):
    notch_pos_list = [int(notch*src.shape[0]/sampling_rate) for notch in notch_list]
    filter_notch_f = makeNotchFilter(notch_pos_list, sigma_list, src.shape[0])
    src_f = fftpack.fft(src, axis=0)
    filtered_f = np.multiply(src_f, filter_notch_f)
    dst = fftpack.ifft(filtered_f, axis=0).real.astype(np.float32)
    return dst


def channelNormalize(src):
    assert src.ndim == 2, 'number of dimensions of input must be 2'
    dst = np.zeros_like(src)
    for channel in range(0, src.shape[1]):
        data = src[:, channel]
        v_min = np.min(data)
        v_max = np.max(data)
        scalar = 2 * np.max([np.abs([v_min, v_max])])
        dst[:, channel] = 0.5 + data / scalar
    return dst
