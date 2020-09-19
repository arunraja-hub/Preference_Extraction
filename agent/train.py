r"""General TF-Agents trainer executable.

Runs training on a TFAgent in a specified environment. It is recommended that
the agent be configured using Gin-config and the --gin_file flag, but you
can also import the train function and pass an agent class that you have
configured manually.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf
import gin.tf.external_configurables

import trainer


flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_alias('job-dir', 'root_dir')
flags.DEFINE_multi_string('gin_file', 'configs/ppo.gin',
                          'Paths to the study config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')

FLAGS = flags.FLAGS


def main(_):

  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_v2_behavior()
  tf.compat.v1.enable_resource_variables()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings,
                                      skip_unknown=True)

  root_dir = FLAGS.root_dir
  with gin.unlock_config():
    gin.bind_parameter('%ROOT_DIR', root_dir)

  trainer.train(root_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)
