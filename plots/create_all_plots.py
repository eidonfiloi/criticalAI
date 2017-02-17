import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle


def plot_histogram(data, title="", bin=80, x_label="log weight degree", y_label="log counts", save_to=None, loglog=True):
    # the histogram of the raw_data

    if loglog:
        data_ = np.log10([el for el in data if el > 0.0])
    else:
        data_ = data
    cmap = plt.get_cmap('viridis')
    color = random.choice(cmap.colors[1:])
    binN = 2 * int(math.sqrt(len(data_)))
    binw = (max(data_) - min(data_)) / binN
    bins = np.arange(min(data_), max(data_) + binw, binw)

    plt.figure()
    if loglog:
        plt.hist(data_, bins=bin, log=True, color=color, alpha=0.90)
    else:
        plt.hist(data_, bins=bin, log=False, color=color, alpha=0.90)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)

    if save_to is not None:
        plt.savefig(save_to + '.pdf', format='pdf')
    else:
        plt.show()


def plot_line(data, title="", x_label="log weight degree", y_label="log counts", save_to=None):

    cmap = plt.get_cmap('viridis')
    color = random.choice(cmap.colors)
    plt.figure()
    plt.plot(data[0], data[1], 'o', color=color)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_to is None:
        plt.show()
    else:
        plt.savefig(save_to + '.pdf', format='pdf')


def create_eval_plots(path_, dirs=('ff', 'cnn')):
    evals = []
    for dir in dirs:
        evals += [dir + "/" + fn for fn in os.listdir(path_ + dir) if fn.startswith("eval")]

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

        plt.show()


def create_plots(path_, pattern_, model_name_):
    pickles = [fn for fn in os.listdir(path_) if pattern_ in fn]
    weights = [fn for fn in pickles if fn.startswith("weight")]
    acts = [fn for fn in pickles if fn.startswith("act")]
    node_acts = [fn for fn in pickles if fn.startswith("node_act")]
    sum_acts = [fn for fn in pickles if fn.startswith("sum_act")]

    if len(node_acts) > 0:
        for fn in node_acts:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                for j, d in data.items():
                    l = len(d)
                    logrank = [math.log10(i) for i in range(1, l + 1)]
                    logd = [math.log10(float(el)+1.0) for el in sorted(d, reverse=True)]
                    plot_line([logrank, logd], x_label="log activation rank", title="single node activation frequency",
                              save_to="pdfs/{2}_{0}_node_act_plot_{1}".format(name, j, model_name_))

    if len(acts) > 0:
        for fn in acts:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                for j, d in data.items():
                    l = len(d)
                    logrank = [math.log10(i) for i in range(1, l + 1)]
                    logd = [math.log10(float(el)) for el in d]
                    plot_line([logrank, logd], x_label="log activation rank", title="activation pattern frequency", save_to="pdfs/{2}_{0}_act_plot_{1}".format(name, j, model_name_))

    if len(weights) > 0:
        for fn in weights:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                b = 2 * int(math.sqrt(len(data)))
                plot_histogram(data, bin=b, title="weight distribution", save_to="pdfs/{1}_{0}_weight_plot".format(name, model_name_))

    if len(sum_acts) > 0:
        for fn in sum_acts:
            name = fn.split(".")[0]
            with open(path_ + fn, 'rb') as p:
                data = pickle.load(p)
                for j, d in data.items():
                    plot_histogram(d, title="average activations", save_to="pdfs/{2}_{0}_sum_act_plot_{1}".format(name, j, model_name_), x_label="log avg activations", y_label="log count")

if __name__ == '__main__':
    model_name = "ff"
    model_pattern = "500_400_300_200_200"
    path = "../results/{0}/".format(model_name)
    create_plots(path, model_pattern, model_name)

    # create_eval_plots("../results/")