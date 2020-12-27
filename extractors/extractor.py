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

    def train_single_run(self, xs, ys, repeat_no):
        
        loop_base = (self.num_val+self.num_train)*repeat_no
        train_end = loop_base+self.num_train
        val_end = train_end+self.num_val
        
        xs_train = xs[loop_base:train_end]
        ys_train = ys[loop_base:train_end]
        
        xs_val = xs[train_end:val_end]
        ys_val = ys[train_end:val_end]

        return self.train_single(xs_train, ys_train, xs_val, ys_val)
