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

from extractors.tf_extractor import TfExtractor
from extractors.torch_extractor import TorchExtractor

sys.path.append('agent')

flags.DEFINE_string('data_path', None, 'Path to experience data to extract preferences from')
flags.DEFINE_string('agent_path', None, 'Path to agent to be used to learn preferences')
flags.DEFINE_string('gin_file', "", 'Paths to the study config file.')
FLAGS = flags.FLAGS

def data_pipeline(data_path, env='doom', rebalance=True):
    if data_path[:5] == 'gs://':  # if GCP path
        data = get_data_from_gcp(data_path)
    else:  # for data saved localy as list of trajectories
        data = get_data_from_folder(data_path)
    
    xs, ys = transform_to_x_y(data, env=env)
    if rebalance:
        xs, ys = rebalance_data_to_minority_class(xs, ys)
    
    return xs, ys
    
def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v2.enable_v2_behavior()
    
    gin_file = FLAGS.gin_file
    gin.parse_config_file(gin_file, skip_unknown=True)
    exp_data_path = FLAGS.data_path
    agent_path = FLAGS.agent_path
    
    xs, ys = data_pipeline(exp_data_path)
    
    with gin.unlock_config():
        gin.bind_parameter('%AGENT_DIR', agent_path)
        gin.bind_parameter('%INPUT_SHAPE', xs.shape)
    
    extractor = TorchExtractor()
    #extractor.train(xs, ys)


if __name__ == '__main__':
    flags.mark_flag_as_required('data_path')
    app.run(main)