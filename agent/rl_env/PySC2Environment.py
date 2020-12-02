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
FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS

@gin.configurable
class PySC2Environment(py_environment.PyEnvironment):

    def __init__(self, flatten_action_specs, map_name, agent, agent_race, agent_name, agent2, agent2_race,
                difficulty, bot_build, battle_net_map, feature_screen_size, feature_minimap_size, 
                rgb_screen_size, rgb_minimap_size, action_space, use_feature_units, use_raw_units,
                step_mul, game_steps_per_episode, disable_fog):
        
        # PySC2 environment initialization
        map_inst = maps.get(map_name)
        agent_classes = []
        players = []
        agent_module, agent_name = agent.rsplit(".", 1)
        agent_cls = getattr(importlib.import_module(agent_module), agent_name)
        players.append(sc2_env.Agent(sc2_env.Race[agent_race], agent_name))
        if map_inst.players >= 2:
              if agent2 == "Bot":
                players.append(sc2_env.Bot(sc2_env.Race[agent2_race],
                                           sc2_env.Difficulty[difficulty],
                                           sc2_env.BotBuild[bot_build]))
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            battle_net_map=battle_net_map,
            players=players,
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=feature_screen_size,
                feature_minimap=feature_minimap_size,
                rgb_screen=rgb_screen_size,
                rgb_minimap=rgb_minimap_size,
                action_space=action_space,
                use_feature_units=use_feature_units,
                use_raw_units=use_raw_units),
            step_mul=step_mul,
            game_steps_per_episode=game_steps_per_episode,
            disable_fog=disable_fog,
            visualize=False)
        self.agents = [agent_cls()]

        # Wrapper initialization
        # Observation is feature_screen
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(27, 84, 84), dtype=np.float32, minimum=0, name='observation')

        # Action is move screen (id = 331)
        self.func_id = 331
        arg_sizes = [arg.sizes for arg in self.env.action_spec()[0].functions[self.func_id].args][1]

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
                                  reward=self.timesteps[0].reward)

        return ts.transition(np.array(self.timesteps[0].observation.feature_screen, dtype=np.float32),
                            reward=self.timesteps[0].reward)


def main(unused_argv):
    environment = PySC2Environment()
    utils.validate_py_environment(environment, episodes=3)

@gin.configurable
def pysc2_tf_agents_env(_):
    return PySC2Environment()



if __name__ == "__main__":
    app.run(main)
