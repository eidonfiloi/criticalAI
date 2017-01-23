from dataprovider.cifar import *
import tensorflow as tf

from models.CriticalEchoMirrorNet import *

import sys


cifar = DataLoader()

n_samples = cifar.num_examples

total_batch = int(n_samples / 100)

batch_images = cifar.next_batch(100)

print(batch_images)