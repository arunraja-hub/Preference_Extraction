r"""General TF-Agents evaluator executable.

Runs evaluation on policies generated by a TFAgent.

Evaluations are run every time a new policy checkpoint is saved to the
$ROOT_DIR/train/policy directory. So for this to work, you must be running the
trainer (see trainer.py in this directory) so that there are policies to
evaluate. If there are policies already in the directory, only the latest is
evaluated (and any subsequent ones generated after the evaluator is started).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

from absl import app
from absl import flags
from absl import logging

import gin
import gin.tf
import tensorflow as tf
import gin.tf.external_configurables

import evaluator

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', "configs/dqn.gin",
                          'Paths to the study config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding to pass through.')
FLAGS = flags.FLAGS


def main(_):
  logging.set_verbosity(logging.INFO)
  tf.compat.v1.enable_resource_variables()
  tf.compat.v2.enable_v2_behavior()

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_bindings,
                                      skip_unknown=True)

  root_dir = FLAGS.root_dir

  score_acc = evaluator.WindowedScoreAccumulator()

  with gin.unlock_config():
    gin.bind_parameter('%ROOT_DIR', root_dir)

  agent_evaluator = evaluator.Evaluator(
      root_dir,
      eval_metrics_callback=None)
  agent_evaluator.watch_until()


if __name__ == '__main__':
  flags.mark_flag_as_required('root_dir')
  app.run(main)