from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import gin
import math

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import importlib
import threading

tf.compat.v1.enable_v2_behavior()

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch
from pysc2.lib import actions

from absl import logging

""" Enviroment wrapper based on
    * https://github.com/deepmind/pysc2/blob/master/pysc2/bin/agent.py
    * https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
    * https://github.com/tensorflow/agents/blob/master/docs/tutorials/2_environments_tutorial.ipynb
"""

FLAGS = flags.FLAGS

point_flag.DEFINE_point("feature_screen_size", "84", "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64", "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None, "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None, "Resolution for rendered minimap.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None, "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None, "Name of the agent in replays. Defaults to the class name.")


flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature and rgb observations.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")

flags.DEFINE_bool("disable_fog", True, "Whether to disable Fog of War.")
flags.DEFINE_bool("use_feature_units", False, "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", True, "Whether to include raw units.")
flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.DEFINE_bool("render", False, "Whether to render with pygame.")

flags.mark_flag_as_required("map")

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

@gin.configurable
class PySC2Env(py_environment.PyEnvironment):

    def __init__(self, flatten_action_specs):
        # PySC2 environment initialization
        map_inst = maps.get(FLAGS.map)
        agent_classes = []
        players = []
        agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
        agent_cls = getattr(importlib.import_module(agent_module), agent_name)
        agent_classes.append(agent_cls)
        players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                                     FLAGS.agent_name or agent_name))
        if map_inst.players >= 2:
              if FLAGS.agent2 == "Bot":
                players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                           sc2_env.Difficulty[FLAGS.difficulty],
                                           sc2_env.BotBuild[FLAGS.bot_build]))
        self.env = sc2_env.SC2Env(
            map_name=FLAGS.map,
            battle_net_map=FLAGS.battle_net_map,
            players=players,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=FLAGS.feature_screen_size,
                feature_minimap=FLAGS.feature_minimap_size,
                rgb_screen=FLAGS.rgb_screen_size,
                rgb_minimap=FLAGS.rgb_minimap_size,
                action_space=FLAGS.action_space,
                use_feature_units=FLAGS.use_feature_units,
                use_raw_units=FLAGS.use_raw_units),
            step_mul=FLAGS.step_mul,
            game_steps_per_episode=FLAGS.game_steps_per_episode,
            disable_fog=FLAGS.disable_fog,
            visualize=False)
        self.agents = [agent_cls() for agent_cls in agent_classes]

        # Wrapper initialization
        # Observation is feature_screen
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(27, 84, 84), dtype=np.float32, minimum=0, name='observation')

        # Action is move screen (id = 331)
        self.func_id = 331
        arg_sizes = [arg.sizes for arg in env.action_spec()[0].functions[self.func_id].args][1]

        self.flatten_action_specs = flatten_action_specs
        if flatten_action_specs:
            self.act_space = max(arg_sizes)
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int64, minimum=0, maximum=self.act_space ** 2, name='action')
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=np.array(arg_sizes).shape, dtype=np.int64, minimum=0, maximum=max(arg_sizes), name='action')

    def action_spec(self):
        return self._action_spec


    def observation_spec(self):
        return self._observation_spec


    def _reset(self):
        self.timesteps = self.env.reset()
        for a in self.agents:
            a.reset()
        return ts.restart(np.array(self.timesteps[0].observation.feature_screen, dtype=np.float32))


    def _step(self, action):

        if self.timesteps[0].last():
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        if self.flatten_action_specs:# Un-flattens action
            action = (math.floor(action / self.act_space), self.act_space - (action % self.act_space))

        if int(actions.FUNCTIONS.Move_screen.id) in self.timesteps[0].observation.available_actions:
            action_to_take = [actions.FUNCTIONS.Move_screen("now", action)]
        else:
            action_to_take = [actions.FUNCTIONS.select_army("select")]

        self.timesteps = self.env.step(action_to_take)

        if self.timesteps[0].last():
            return ts.termination(np.array(self.timesteps[0].observation.feature_screen, dtype=np.float32),
                                  reward=self.timesteps[0].observation.reward)

        return ts.transition(np.array(self.timesteps[0].observation.feature_screen, dtype=np.float32),
                            reward=self.timesteps[0].reward)


def main(unused_argv):
    environment = PySC2Env()
    utils.validate_py_environment(environment, episodes=3)

@gin.configurable
def pysc2_tf_agents_env(_):
    return PySC2Env()



if __name__ == "__main__":
    app.run(main)
