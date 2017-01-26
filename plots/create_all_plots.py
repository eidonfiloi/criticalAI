import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


def plot_histogram(data, title="", x_label="log weight degree", y_label="log counts", save_to=None, loglog=True):
    # the histogram of the raw_data

    if loglog:
        data_ = np.log10(data)
    else:
        data_ = data
    cmap = plt.get_cmap('viridis')
    color = random.choice(cmap.colors)
    binN = 2 * int(math.sqrt(len(data_)))
    binw = (max(data_) - min(data_)) / binN
    bins = np.arange(min(data_), max(data_) + binw, binw)

    plt.figure()
    if loglog:
        plt.hist(data_, bins=bins, log=True, color=color, alpha=0.90)
    else:
        plt.hist(data_, bins=bins, log=False, color=color, alpha=0.90)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_line(data, title="", x_label="log weight degree", y_label="log counts", save_to=None):

    cmap = plt.get_cmap('viridis')
    color = random.choice(cmap.colors)
    plt.figure()
    plt.plot(data[0], data[1], color=color)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to)


def create_eval_plots(path_, dirs=('ff', 'cnn')):
    evals = []
    for dir in dirs:
        evals += [fn for fn in os.listdir(path_ + dir) if fn.startswith("eval")]

    if len(evals) > 0:
        datas = {}
        for fn in evals:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                datas[name] = data

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        for k, d in datas.items():
            ax1.plot(d, label=k)

        colormap = plt.get_cmap('viridis')
        colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
        for i, j in enumerate(ax1.lines):
            j.set_color(colors[i])

        ax1.legend(loc=2)


def create_plots(path_, pattern_):
    pickles = [fn for fn in os.listdir(path_) if pattern_ in fn]
    weights = [fn for fn in pickles if fn.startswith("weight")]
    acts = [fn for fn in pickles if fn.startswith("act")]
    sum_acts = [fn for fn in pickles if fn.startswith("sum_act")]

    if len(acts) > 0:
        for fn in acts:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                for j, d in data.items():
                    l = len(d)
                    logrank = [math.log10(i) for i in range(1, l + 1)]
                    logd = [math.log10(float(el)) for el in d]
                    plot_line([logrank, logd], save_to="{0}_act_plot_{1}".format(name, j))

    if len(weights) > 0:
        for fn in weights:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                plot_histogram(data, title="weight distribution", save_to="{0}_weight_plot".format(name))

    if len(sum_acts) > 0:
        for fn in sum_acts:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                for j, d in data.items():
                    plot_histogram(d, title="average activations", save_to="{0}_sum_act_plot_{1}".format(name, j), x_label="log avg activations", y_label="log count")

if __name__ == '__main__':
    path = "../results/ae/"
    model_pattern = "600"
    create_plots(path, model_pattern)