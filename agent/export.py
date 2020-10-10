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
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', "",
                          'Paths to the study config files.')
flags.DEFINE_bool('add_root_dir_gin_file', False, "")
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')
FLAGS = flags.FLAGS

def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_resource_variables()
    tf.compat.v2.enable_v2_behavior()
    
    gin_file = FLAGS.gin_file
    if FLAGS.add_root_dir_gin_file:
        gin_file.append(os.path.join(FLAGS.root_dir, "train/operative_config-0.gin"))
        
    gin.parse_config_files_and_bindings(gin_file, FLAGS.gin_bindings, skip_unknown=True)
    
    root_dir = FLAGS.root_dir
    
    with gin.unlock_config():
        gin.bind_parameter('%ROOT_DIR', root_dir)

    agent_exporter = exporter.Exporter(root_dir)

if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)