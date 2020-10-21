"""
    Main launcher for preference extraction experiments
"""
import sys

from absl import app
from absl import flags
from absl import logging

import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

from data_getter import get_data_from_gcp, get_data_from_folder
from data_processing import transform_to_x_y, rebalance_data_to_minority_class

from tf_extractor import TfExtractor
from torch_extractor import TorchExtractor

flags.DEFINE_multi_string('gin_file', '', 'Paths to the study config file.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')
FLAGS = flags.FLAGS

@gin.configurable
def data_pipeline(data_path, env, rebalance):
    if data_path[:5] == 'gs://':  # if GCP path
        data = get_data_from_gcp(data_path)
    else:  # for data saved localy as list of trajectories
        data = get_data_from_folder(data_path)
    
    xs, ys = transform_to_x_y(data, env=env.lower())
    if rebalance:
        xs, ys = rebalance_data_to_minority_class(xs, ys)
    
    return xs, ys
    
def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v2.enable_v2_behavior()
    
    gin_file = FLAGS.gin_file
    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_bindings, skip_unknown=True)
    
    xs, ys = data_pipeline()
    
    with gin.unlock_config():
        gin.bind_parameter('%INPUT_SHAPE', xs.shape[1:])
    
    if gin_file[0].split('/')[-1] == 'tf.gin':
        extractor = TfExtractor()
    elif gin_file[0].split('/')[-1] == 'torch.gin':
        extractor = TorchExtractor()
    else:
        print('Error! Name of gin config does not suggest an extractor')
        extractor = None
    
    extractor.train(xs, ys)

if __name__ == '__main__':
    flags.mark_flag_as_required('gin_file')
    app.run(main)