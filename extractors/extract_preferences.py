"""
    Main launcher for preference extraction experiments
"""
import os

from absl import app
from absl import flags
from absl import logging

import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

from data_getter import get_data_from_file, get_data_from_folder
from data_processing import transform_to_x_y, rebalance_data_to_minority_class

from tf_extractor import TfExtractor
from torch_extractor import TorchExtractor

flags.DEFINE_multi_string('gin_file', None, 'Paths to the study config file.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')
FLAGS = flags.FLAGS

@gin.configurable
def data_pipeline(data_path, from_file=True, env='doom', rebalance=True):
    if from_file:
        data = get_data_from_file(data_path)
    else: # for data as list of trajectory files
        data = get_data_from_folder(data_path)

    xs, ys = transform_to_x_y(data, env=env)
    if rebalance:
        xs, ys = rebalance_data_to_minority_class(xs, ys)
    
    return xs, ys

@gin.configurable
def extractor_type(extractor):
    return extractor()

def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v2.enable_v2_behavior()
    
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings, skip_unknown=True)
    
    xs, ys = data_pipeline()
    
    with gin.unlock_config():
        gin.bind_parameter('%INPUT_SHAPE', xs.shape[1:])
    
    extractor = extractor_type()
    extractor.train(xs, ys)

if __name__ == '__main__':
    flags.mark_flag_as_required('gin_file')
    app.run(main)