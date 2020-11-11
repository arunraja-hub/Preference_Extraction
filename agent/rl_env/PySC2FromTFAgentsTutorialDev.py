from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np

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
flags.DEFINE_bool("render", False, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "84",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", True,
                  "Whether to include raw units.")
flags.DEFINE_bool("disable_fog", True, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")

flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", True, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.mark_flag_as_required("map")

from pysc2.lib import features

PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))

class PySC2EnvReduced(py_environment.PyEnvironment):
    
    def __init__(self):
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
        env = sc2_env.SC2Env(
            map_name='MoveToBeacon',
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
        self.env = env
        self.agents = [agent_cls() for agent_cls in agent_classes]
        observation_spec = env.observation_spec()
        action_spec = env.action_spec()
        for agent, obs_spec, act_spec in zip(self.agents, observation_spec, action_spec):
            agent.setup(obs_spec, act_spec)
        self.timesteps = env.reset()
        for a in self.agents:
            a.reset()
        
        # Wrapper initialization
        self._episode_ended = False
        
        # Observation is feature_screen
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(27, 84, 84), dtype=np.int32, minimum=0, name='observation')
        
        # Action is move_camera (id = 1)
        self.func_id = 1
        arg = [arg.sizes for arg in action_spec[0].functions[self.func_id].args][0] # fn has only one arg
        self._action_spec = array_spec.BoundedArraySpec(
           shape=np.array(arg).shape, dtype=np.int32, minimum=min(arg), maximum=max(arg), name='action')
        

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        return ts.restart(np.array(self.timesteps[0].observation.feature_screen, dtype=np.int32))
    

    
    def _step(self, action):
        
        obs = self.timesteps[0]
        
        if obs.last():
            self._episode_ended = True
            return ts.termination(np.array(obs.observation.feature_screen, dtype=np.int32),
                                  obs.observation.score_cumulative["score"])
        
        # Scripted move to beacon agent
        # see https://github.com/deepmind/pysc2/blob/05b28ef0d85aa5eef811bc49ff4c0cbe496c0adb/pysc2/agents/scripted_agent.py#L40
        if actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not beacon:
                action_to_take = [actions.FUNCTIONS.no_op()]
            beacon_center = np.mean(beacon, axis=0).round()
            # rand_move = np.random.randint(low=0, high=84, size=(2,))
            action_to_take = [actions.FUNCTIONS.Move_screen("now", beacon_center)] 
        else:
            action_to_take = [actions.FUNCTIONS.select_army("select")]
        
        print(obs.reward, obs.observation.score_cumulative["score"])
        print(action_to_take)
        
        self.timesteps = self.env.step(action_to_take)
        
        return ts.transition(np.array(obs.observation.feature_screen, dtype=np.int32), 
                             reward=obs.reward, discount=obs.discount)


def main(unused_argv):
    environment = PySC2EnvReduced()
    utils.validate_py_environment(environment, episodes=1)

    
if __name__ == "__main__":
    app.run(main)
