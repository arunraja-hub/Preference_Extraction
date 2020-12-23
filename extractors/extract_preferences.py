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

import hypertune
import numpy as np

from data_getter import get_data_from_file, get_data_from_folder
from data_processing import transform_to_x_y, rebalance_data_to_minority_class

from tf_extractor import TfExtractor
from torch_extractor import TorchExtractor

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'root_dir')
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
def train_and_report_metrics(xs, ys, num_repeat, extractor_class):
    """
        Trains the model multiple times with the same parameters and returns the average metrics
    """

    all_val_auc = []
    all_val_accuracy = []

    for i in range(num_repeat):
        single_train_metrics = extractor_class().train_single_shuffle(xs, ys)

        all_val_auc.append(single_train_metrics['val_auc'])
        all_val_accuracy.append(single_train_metrics['val_accuracy'])

    metrics = {
        "mean_val_auc": np.mean(all_val_auc),
        "mean_val_accuracy": np.mean(all_val_accuracy),
        "val_auc_std": np.std(all_val_auc),
        "val_accuracy_std": np.std(all_val_accuracy)
    }

    print(metrics, flush=True)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='mean_val_auc',
        metric_value=metrics['mean_val_auc'])

    return metrics

def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v2.enable_v2_behavior()
    
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings, skip_unknown=True)
    
    xs, ys = data_pipeline()
    
    with gin.unlock_config():
        gin.bind_parameter('%INPUT_SHAPE', xs.shape[1:])

    train_and_report_metrics(xs, ys)

    config_filename = os.path.join(FLAGS.root_dir, 'operative_config-final.gin')
    with tf.io.gfile.GFile(config_filename, 'wb') as f:
        f.write(gin.operative_config_str())

if __name__ == '__main__':
    flags.mark_flag_as_required('gin_file')
    app.run(main)