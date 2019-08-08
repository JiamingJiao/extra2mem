import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def plot(data, fig_size=(15, 20), **kwargs):
    channels_num = data.shape[1]
    fig = plt.figure(figsize=fig_size)
    label_position = (1.05*data.shape[0], 0)
    for i in range(channels_num):
        plt.subplot(channels_num, 1, i+1)
        plt.plot(data[:, i], **kwargs)
        plt.text(*label_position, '%d'%(i+1))
    plt.close()
    return fig


def makeMovie(src, color_map, interval):
    if src.ndim==4:
        data = src[..., 0]
    elif src.ndim==3:
        data = src
    v_min = np.min(data)
    v_max = np.max(data)
    fig = plt.figure()
    sequence = []
    for img in data:
        frame = plt.imshow(img, vmin=v_min, vmax=v_max, cmap=color_map)
        sequence.append([frame])
    plt.colorbar()
    video_writer = animation.ArtistAnimation(fig, sequence, interval=interval)
    return video_writer
