'''
loads all the data from cifar into memory.
some help from this insightful blog post: http://aidiary.hatenablog.com/entry/20151014/1444827123
assumes data is in cifar-10-batches-py, and cpkl'ed python version of data is downloaded/unzipped there.

0 - airplane
1 - automobile
2 - bird
3 - cat
4 - deer
5 - dog
6 - frog
7 - horse
8 - ship
9 - truck

'''
# %matplotlib inline
import sklearn.preprocessing as prep

import collections

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random as random
from dataprovider.DataSet import *
from tensorflow.contrib.learn.python.learn.datasets import base

Datasets = collections.namedtuple('Datasets', ['train', 'test'])


class CifarData(object):
    """
     Cifar data set
    """

    def __init__(self, data_dir="raw_data/cifar-10-batches-py", reshape=True):
        self.data_dir = data_dir
        train_datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_datafiles = ['test_batch']

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        def unpickle(f):
            fo = open(f, 'rb')
            d = pickle.load(fo, encoding="latin1")
            fo.close()
            return d

        for f in train_datafiles:
            d = unpickle(self.data_dir + '/' + f)
            data = d["data"]
            labels = np.array(d["labels"])
            nsamples = len(data)
            # targets = np.where(labels == target_label)[0]
            for idx in range(nsamples):
                train_data.append(data[idx].reshape(3, 32, 32).transpose(1, 2, 0))
                train_labels.append(labels[idx])

        for f in test_datafiles:
            d = unpickle(self.data_dir + '/' + f)
            data = d["data"]
            labels = np.array(d["labels"])
            nsamples = len(data)
            # targets = np.where(labels == target_label)[0]
            for idx in range(nsamples):
                test_data.append(data[idx].reshape(3, 32, 32).transpose(1, 2, 0))
                test_labels.append(labels[idx])

        if reshape:
            train_ = self.rgb2gray(np.array(train_data, dtype=np.float32))
            test_ = self.rgb2gray(np.array(test_data, dtype=np.float32))

        else:
            train_ = np.array(train_data, dtype=np.float32)
            test_ = np.array(test_data, dtype=np.float32)

        train_labels_ = self.dense_to_one_hot(np.array(train_labels, dtype=np.uint8))
        test_labels_ = self.dense_to_one_hot(np.array(test_labels, dtype=np.uint8))

        self.train = DataSet(train_, train_labels_, dtype=dtypes.float32, reshape=reshape)
        self.test = DataSet(test_, test_labels_, dtype=dtypes.float32, reshape=reshape)

    def read_data_sets(self):

        return base.Datasets(train=self.train, test=self.test, validation=None)

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [[0.299], [0.587], [0.114]])

    def dense_to_one_hot(self, labels_dense, num_classes=10):
        """Convert class labels from scalars to one-hot vectors."""
        num_labels = labels_dense.shape[0]
        index_offset = numpy.arange(num_labels) * num_classes
        labels_one_hot = numpy.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot


class DataLoader():
    def __init__(self, batch_size=100, target_label=6, data_dir="raw_data/cifar-10-batches-py"):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_label = target_label

        datafiles = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        test_datafiles = ['test_batch']

        def unpickle(f):
            fo = open(f, 'rb')
            d = pickle.load(fo, encoding="latin1")
            fo.close()
            return d

        self.train_data = []
        self.test_data = []

        for f in datafiles:
            d = unpickle(self.data_dir + '/' + f)
            data = d["data"]
            labels = np.array(d["labels"])
            nsamples = len(data)
            # targets = np.where(labels == target_label)[0]
            for idx in range(nsamples):
                self.train_data.append(data[idx].reshape(3, 32, 32).transpose(1, 2, 0))

        for f in test_datafiles:
            d = unpickle(self.data_dir + '/' + f)
            data = d["data"]
            labels = np.array(d["labels"])
            nsamples = len(data)
            # targets = np.where(labels == target_label)[0]
            for idx in range(nsamples):
                self.test_data.append(data[idx].reshape(3, 32, 32).transpose(1, 2, 0))

        self.train_data = np.array(self.train_data, dtype=np.float32)
        self.test_data = np.array(self.train_data, dtype=np.float32)

        self.train_data /= 255.0

        self.num_examples = len(self.train_data)

        self.pointer = 0

        self.shuffle_data()

    def show_random_image(self):
        pos = 1
        for i in range(10):
            for j in range(10):
                plt.subplot(10, 10, pos)
                img = random.choice(self.train_data)
                # (channel, row, column) => (row, column, channel)
                plt.imshow(np.clip(img, 0.0, 1.0), interpolation='none')
                plt.axis('off')
                pos += 1
        plt.show()

    def show_image(self, image):
        '''
        image is in [height width depth]
        '''
        plt.subplot(1, 1, 1)
        plt.imshow(np.clip(image, 0.0, 1.0), interpolation='none')
        plt.axis('off')
        plt.show()

    def next_batch(self, batch_size, distorted=False, flatten=False):
        self.pointer += batch_size
        if self.pointer >= self.num_examples:
            self.pointer = 0
        result = []

        def random_flip(x):
            if np.random.rand(1)[0] > 0.5:
                return np.fliplr(x)
            return x

        for data in self.train_data[self.pointer:self.pointer + batch_size]:
            result.append(random_flip(data))
        if distorted:
            raw_result = self.distort_batch(np.array(result, dtype=np.float32))
        else:
            raw_result = np.array(result, dtype=np.float32)

        if flatten:
            return raw_result
        else:
            return raw_result

    def distort_batch(self, batch):
        batch_size = len(batch)
        row_distort = np.random.randint(0, 3, batch_size)
        col_distort = np.random.randint(0, 3, batch_size)
        result = np.zeros(shape=(batch_size, 30, 30, 3), dtype=np.float32)
        for i in range(batch_size):
            result[i, :, :, :] = batch[i, row_distort[i]:row_distort[i] + 30, col_distort[i]:col_distort[i] + 30, :]
        return result

    def shuffle_data(self):
        self.train_data = np.random.permutation(self.train_data)
