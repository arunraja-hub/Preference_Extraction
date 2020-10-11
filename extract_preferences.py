"""
    Main launcher for preference extraction experiments
"""
import sys

from absl import app
from absl import flags

import tensorflow as tf

from data_getter import get_data_from_gcp, get_data_from_folder
from data_processing import transform_to_x_y, rebalance_data_to_minority_class

sys.path.append('agent')

flags.DEFINE_string('data_path', None, 'Path to experience data to extract preferences from')
flags.DEFINE_string('agent_path', None, 'Path to agent to be used to learn preferences')
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

def agent_pipeline(agent_path):
    model = tf.keras.models.load_model(agent_path)
    model.summary()
    
    
def main(_):
    exp_data_path = FLAGS.data_path
    agent_path = FLAGS.agent_path
    
    # xs, ys = data_pipeline(exp_data_path)
    agent_pipeline(agent_path)
    
if __name__ == '__main__':
    flags.mark_flag_as_required('data_path')
    app.run(main)