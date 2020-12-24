import gin
import gin.tf
import gin.tf.external_configurables

import numpy as np

def get_layer_sizes(first_size, last_size, num_layers):
    return np.linspace(first_size, last_size, num_layers, dtype=np.int32)

@gin.configurable
class Extractor(object):
    def __init__(self, num_train, num_val):
        self.num_train = num_train
        self.num_val = num_val

    def train_single_shuffle(self, xs, ys):
        """
        Trains the model and reruns the logs of the best epoch.
        Randomly splits the train and val data before training.
        """

        randomize = np.arange(len(xs))
        np.random.shuffle(randomize)
        xs = xs[randomize]
        ys = ys[randomize]

        xs_train = xs[:self.num_train]
        ys_train = ys[:self.num_train]
        xs_val = xs[self.num_train:self.num_train + self.num_val]
        ys_val = ys[self.num_train:self.num_train + self.num_val]

        return self.train_single(xs_train, ys_train, xs_val, ys_val)
