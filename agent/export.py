import os

from absl import app
from absl import flags
from absl import logging

import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

import exporter

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for reading logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', "", 'Paths to the study config files.')
flags.DEFINE_string('checkpoint', None, 'Checkpoint at which export agent')
flags.DEFINE_string('collect_data', None, 'Number of experience data points to collect')
FLAGS = flags.FLAGS

def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v2.enable_v2_behavior()
    
    gin_file = FLAGS.gin_file
    gin_file.append(os.path.join(FLAGS.root_dir, "train/operative_config-0.gin"))
    gin.parse_config_files_and_bindings(gin_file, None, skip_unknown=True)
    root_dir = FLAGS.root_dir
    checkpoint_no = FLAGS.checkpoint
    collect_data = FLAGS.collect_data
    if collect_data is None:
        collect_data = 0
    else:
        collect_data = int(collect_data)
    
    with gin.unlock_config():
        gin.bind_parameter('%ROOT_DIR', root_dir)

    agent_exporter = exporter.Exporter(root_dir, checkpoint_no, collect_data=collect_data, verify_restore_success=True)

if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)