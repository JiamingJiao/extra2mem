import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import numpy as np


def plot(data, fig_size=(15, 20)):
    channels_num = data.shape[1]
    fig = plt.figure(figsize=fig_size)
    for i in range(channels_num):
        plt.subplot(channels_num, 1, i+1)
        plt.plot(data[:, i])
    return fig


def showMovie(src, color_map, interval):
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
    HTML(video_writer.to_html5_video())
