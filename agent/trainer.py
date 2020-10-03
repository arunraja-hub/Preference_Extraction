r"""Generic TF-Agents training function.

Runs training on a TFAgent in a specified environment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time

from absl import logging

import gin
import gin.tf
from six.moves import range
import tensorflow as tf
import hypertune

from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.agents.ppo import ppo_kl_penalty_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import tf_py_metric
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import environment_specs
from rl_env import DoomEnviroment

ON_POLICY_AGENTS = (
    ppo_agent.PPOAgent,
    ppo_clip_agent.PPOClipAgent,
    ppo_kl_penalty_agent.PPOKLPenaltyAgent,
    reinforce_agent.ReinforceAgent,
)

PPO_AGENTS = (
    ppo_agent.PPOAgent,
    ppo_clip_agent.PPOClipAgent,
    ppo_kl_penalty_agent.PPOKLPenaltyAgent,
)

REINFORCE_AGENTS = (
    reinforce_agent.ReinforceAgent,
)

# Loss value that is considered too high and training will be terminated.
MAX_LOSS = 1e9
# How many steps does the loss have to be diverged for (too high, inf, nan)
# after the training terminates. This should prevent termination on short loss
# spikes.
TERMINATE_AFTER_DIVERGED_LOSS_STEPS = 100


def cleanup_checkpoints(checkpoint_dir):
  checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
  if checkpoint_state is None:
    return
  for checkpoint_path in checkpoint_state.all_model_checkpoint_paths:
    tf.compat.v1.train.remove_checkpoint(checkpoint_path)


@gin.configurable
def train(
    root_dir,
    env_load_fn=suite_gym.load,
    env_name='CartPole-v0',
    env_name_eval=None,
    num_parallel_environments=1,
    agent_class=None,
    initial_collect_random=True,
    initial_collect_driver_class=None,
    collect_driver_class=None,
    num_global_steps=100000,
    train_steps_per_iteration=1,
    clear_rb_after_train_steps=None,  # Defaults to True for ON_POLICY_AGENTS
    train_metrics=None,
    # Params for eval
    run_eval=False,
    num_eval_episodes=30,
    eval_interval=1000,
    eval_metrics_callback=None,
    # Params for checkpoints, summaries, and logging
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=20000,
    keep_rb_checkpoint=False,
    train_sequence_length=1,
    log_interval=1000,
    summary_interval=1000,
    summaries_flush_secs=10,
    early_termination_fn=None,
    env_metric_factories=None):

  if not agent_class:
    raise ValueError(
        'The `agent_class` parameter of trainer.train must be set.')

  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  saved_model_dir = os.path.join(root_dir, 'policy_saved_model')
  if not tf.io.gfile.exists(saved_model_dir):
    tf.io.gfile.makedirs(saved_model_dir)

  train_summary_writer = tf.compat.v2.summary.create_file_writer(
      train_dir, flush_millis=summaries_flush_secs * 1000)
  train_summary_writer.set_as_default()

  def make_possibly_parallel_environment(env_name_):
    """Returns a function creating env_name_, possibly a parallel one."""
    if num_parallel_environments == 1:
      return env_load_fn(env_name_)
    else:
      return parallel_py_environment.ParallelPyEnvironment(
          [lambda: env_load_fn(env_name_)] * num_parallel_environments)

  def make_tf_py_envs(env):
    """Convert env to tf if needed."""
    if isinstance(env, py_environment.PyEnvironment):
      tf_env = tf_py_environment.TFPyEnvironment(env)
      py_env = env
    else:
      tf_env = env
      py_env = None  # Can't generically convert to PyEnvironment.
    return tf_env, py_env

  eval_py_env = None
  if run_eval:
    if env_name_eval is None: env_name_eval = env_name
    eval_env = make_possibly_parallel_environment(env_name_eval)
    eval_tf_env, eval_py_env = make_tf_py_envs(eval_env)

    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    eval_metrics = [
        tf_metrics.AverageReturnMetric(batch_size=eval_tf_env.batch_size,
                                       buffer_size=num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=eval_tf_env.batch_size,
                                              buffer_size=num_eval_episodes),
    ]

  global_step = tf.compat.v1.train.get_or_create_global_step()
  with tf.compat.v2.summary.record_if(
      lambda: tf.math.equal(global_step % summary_interval, 0)):
    env = make_possibly_parallel_environment(env_name)
    tf_env, py_env = make_tf_py_envs(env)

    environment_specs.set_observation_spec(tf_env.observation_spec())
    environment_specs.set_action_spec(tf_env.action_spec())

    # Agent params configured with gin.
    agent = agent_class(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        train_step_counter=global_step)
    agent.initialize()

    if clear_rb_after_train_steps is None:
      # Default is to clear RB for ON_POLICY_AGENTS, only.
      clear_rb_after_train_steps = isinstance(agent, ON_POLICY_AGENTS)

    if run_eval:
      eval_policy = greedy_policy.GreedyPolicy(agent.policy)

    if not train_metrics:
      train_metrics = [
          tf_metrics.NumberOfEpisodes(),
          tf_metrics.EnvironmentSteps(),
          tf_metrics.AverageReturnMetric(
              batch_size=tf_env.batch_size, buffer_size=log_interval*tf_env.batch_size),
          tf_metrics.AverageEpisodeLengthMetric(
              batch_size=tf_env.batch_size, buffer_size=log_interval*tf_env.batch_size),
      ]
    else:
      train_metrics = list(train_metrics)

    if env_metric_factories:
      for metric_factory in env_metric_factories:
        py_metric = metric_factory(environment=py_env)
        train_metrics.append(tf_py_metric.TFPyMetric(py_metric))

    logging.info('Allocating replay buffer ...')
    # Add to replay buffer and other agent specific observers.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec)
    logging.info('RB capacity: %i', replay_buffer.capacity)
    agent_observers = [replay_buffer.add_batch]
    initial_collect_policy = agent.collect_policy
    if initial_collect_random:
      initial_collect_policy = random_tf_policy.RandomTFPolicy(
          tf_env.time_step_spec(),
          tf_env.action_spec(),
          info_spec=agent.collect_policy.info_spec)

    collect_policy = agent.collect_policy

    collect_driver = collect_driver_class(
        tf_env, collect_policy, observers=agent_observers + train_metrics)

    rb_ckpt_dir = os.path.join(train_dir, 'replay_buffer')

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        max_to_keep=1,
        agent=agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        max_to_keep=None,
        policy=agent.policy,
        global_step=global_step)
    rb_checkpointer = common.Checkpointer(
        ckpt_dir=rb_ckpt_dir, max_to_keep=1, replay_buffer=replay_buffer)

    saved_model = policy_saver.PolicySaver(
        greedy_policy.GreedyPolicy(agent.policy), train_step=global_step)

    train_checkpointer.initialize_or_restore()
    rb_checkpointer.initialize_or_restore()

    collect_driver.run = common.function(collect_driver.run)
    agent.train = common.function(agent.train)

    if not rb_checkpointer.checkpoint_exists:
      logging.info('Performing initial collection ...')
      common.function(
          initial_collect_driver_class(
              tf_env,
              initial_collect_policy,
              observers=agent_observers + train_metrics).run)()

      if run_eval:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)

    # This is only used for PPO Agents.
    # The dataset is repeated for `train_steps_per_iteration` which represents
    # the number of epochs we loop through during training.
    def get_data_iter_repeated(replay_buffer):
      dataset = replay_buffer.as_dataset(
          sample_batch_size=num_parallel_environments,
          num_steps=train_sequence_length + 1,
          num_parallel_calls=3,
          single_deterministic_pass=True).repeat(train_steps_per_iteration)
      if len([1 for _ in dataset]) == 0:
             logging.warning('PPO Agent replay buffer as dataset is empty')
      return iter(dataset)

    # For off policy agents, one iterator is created for the entire training
    # process. This is different from PPO agents whose iterators are reset
    # in the training loop.
    if not isinstance(agent, ON_POLICY_AGENTS):
      dataset = replay_buffer.as_dataset(
          num_parallel_calls=3, num_steps=train_sequence_length+1).prefetch(3)
      iterator = iter(dataset)

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)
    timed_at_step = global_step.numpy()
    time_acc = 0

    def save_policy(global_step_value):
      """Saves policy using both checkpoint saver and saved model."""
      policy_checkpointer.save(global_step=global_step_value)
      saved_model_path = os.path.join(
          saved_model_dir, 'policy_' + ('%d' % global_step_value).zfill(8))
      saved_model.save(saved_model_path)

    if global_step.numpy() == 0:
      # Save an initial checkpoint so the evaluator runs for global_step=0.
      save_policy(global_step.numpy())

    @common.function
    def train_step(data_iterator):
      experience = next(data_iterator)[0]
      return agent.train(experience)

    @common.function
    def train_with_gather_all():
      return agent.train(replay_buffer.gather_all())

    if not early_termination_fn:
      early_termination_fn = lambda: False

    loss_diverged = False
    # How many consecutive steps was loss diverged for.
    loss_divergence_counter = 0

    # Save operative config as late as possible to include used configurables.
    if global_step.numpy() == 0:
      config_filename = os.path.join(
          train_dir, 'operative_config-{}.gin'.format(global_step.numpy()))
      with tf.io.gfile.GFile(config_filename, 'wb') as f:
        f.write(gin.operative_config_str())

    logging.info('Training ...')
    while (global_step.numpy() <= num_global_steps and
           not early_termination_fn()):
      # Collect and train.
      start_time = time.time()
      time_step, policy_state = collect_driver.run(
          time_step=time_step, policy_state=policy_state)
      if isinstance(agent, PPO_AGENTS):
        iterator = get_data_iter_repeated(replay_buffer)
      for _ in range(train_steps_per_iteration):
        if isinstance(agent, REINFORCE_AGENTS):
          total_loss = train_with_gather_all()
        else:
          total_loss = train_step(iterator)
      total_loss = total_loss.loss

      # Check for exploding losses.
      if (math.isnan(total_loss) or math.isinf(total_loss) or
          total_loss > MAX_LOSS):
        loss_divergence_counter += 1
        if loss_divergence_counter > TERMINATE_AFTER_DIVERGED_LOSS_STEPS:
          loss_diverged = True
          break
      else:
        loss_divergence_counter = 0

      if clear_rb_after_train_steps:
        replay_buffer.clear()
      time_acc += time.time() - start_time

      if global_step.numpy() % log_interval == 0:
        logging.info('step = %d, loss = %f', global_step.numpy(), total_loss)
        steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
        logging.info('%.3f steps/sec', steps_per_sec)
        tf.compat.v2.summary.scalar(
            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
        timed_at_step = global_step.numpy()
        time_acc = 0

      for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=global_step, step_metrics=train_metrics[:2])

        if global_step.numpy() % log_interval == 0:
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=train_metric.name,
                metric_value=train_metric.result(),
                global_step=global_step)

      if global_step.numpy() % train_checkpoint_interval == 0:
        train_checkpointer.save(global_step=global_step.numpy())

      if global_step.numpy() % policy_checkpoint_interval == 0:
        save_policy(global_step.numpy())

      if global_step.numpy() % rb_checkpoint_interval == 0:
        rb_checkpointer.save(global_step=global_step.numpy())

      if run_eval and global_step.numpy() % eval_interval == 0:
        results = metric_utils.eager_compute(
            eval_metrics,
            eval_tf_env,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=eval_summary_writer,
            summary_prefix='Metrics',
        )
        if eval_metrics_callback is not None:
          eval_metrics_callback(results, global_step.numpy())
        metric_utils.log_metrics(eval_metrics)

  if not keep_rb_checkpoint:
    cleanup_checkpoints(rb_ckpt_dir)

  if py_env:
    py_env.close()
  if eval_py_env:
    eval_py_env.close()

  # Save final operative config that will also have all configurables used in
  # the training loop for the first time.
  config_filename = os.path.join(train_dir, 'operative_config-final.gin')
  with tf.io.gfile.GFile(config_filename, 'wb') as f:
    f.write(gin.operative_config_str())

  if loss_diverged:
    # Raise an error at the very end after the cleanup.
    raise ValueError('Loss diverged to {} at step {}, terminating.'.format(
        total_loss, global_step.numpy()))