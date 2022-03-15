import numpy as np
import matplotlib.pyplot as plt


FONTSIZE = 15
LABELS_SIZE = 12
TITLE_SIZE = 16

SAVE_FIG = False


def plot_stats(cpu, gpu, x_label, y_label, title):
    r"""
    Plots average execution times from the three input dicts.

    Parameters
    ----------
    cpu: dict
        First dict with key/values as those returned from the function 'read_from_file()'.
    gpu: dict
        Second dict with key/values as those returned from the function 'read_from_file()'.
    x_label: str
        X axis label.
    y_label: str
        Y axis label.
    title: str
        Title of the plot.
    """
    fig, ax = plt.subplots(figsize=(19.20, 10.80))
    labels = ['Grayscale', 'Gamma', 'Gradients', 'Magnitude/Direction', 'HOG']
    x = np.arange(5)
    width = 0.2  		# width of the bars
    
    r1 = ax.bar(x - width, list(cpu.values())[1:], width, label=f"{cpu['device']} execution time")
    r2 = ax.bar(x, list(gpu.values())[1:], width, label=f"{gpu['device']} execution time")

    ax.set_xticks(x - (width/2), labels, fontsize=FONTSIZE)
    ax.legend(fontsize=LABELS_SIZE)

    ax.set_xlabel(x_label, fontsize=FONTSIZE)
    ax.set_ylabel(y_label, fontsize=FONTSIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)

    ax.bar_label(r1, padding=12, fontsize=LABELS_SIZE)
    ax.bar_label(r2, padding=12, fontsize=LABELS_SIZE)
    fig.tight_layout()

    if SAVE_FIG:
        plt.savefig('timings_cpu-gpu.png', format='png', dpi=600)
    plt.show()


def read_from_file(filename, device):
    r"""
    Reads stats from timing file.

    Parameters
    ----------
    filename: str
        Name of the file to read.
    device: str
        Device for which to compute execution times.

    Returns
    -------
    dct: dict
        Dict containing as keys/values \
            |- {device: d} : d is the device to which execution time refers.\
            |- {grayscale: t} : t is the average execution time for grayscale conversion.\
            |- {gamma: t} : t is the average execution time for gamma correction.\
            |- {gradients: t} : t is the average execution time for gradients computation.\
            |- {magdir: t} : t is the average execution time for magnitude and direction computation.\
            |- {hog: t} : t is the average execution time for HOG computation.
    """
    dct = {'device': device}
    with open(filename, 'r') as f:
        data = f.read()
    timings = data.split("###")
    grayscale = 0
    gamma = 0
    gradients = 0
    magdir = 0
    hog = 0
    for timing in timings:
        stats = timing.split('\n')
        stats = list(filter(lambda x: x!='', stats))
        if len(stats) == 0:
            continue
        grayscale += float(stats[1])
        gamma += float(stats[3])
        gradients += float(stats[5])
        magdir += float(stats[7])
        hog += float(stats[9])
    trunc = 5
    dct['grayscale'] = round(grayscale, trunc)
    dct['gamma'] = round(gamma, trunc)
    dct['gradients'] = round(gradients, trunc)
    dct['magdir'] = round(magdir, trunc)
    dct['hog'] = round(hog, trunc)
    return dct



if __name__=='__main__':
    cpu = read_from_file('video_timing_cpu.txt', 'cpu')
    gpu = read_from_file('video_timing_gpu.txt', 'gpu')
    plot_stats(cpu, gpu, 'Steps of the algorithm', 'Execution time (s)', 'Execution time on test videoclip')