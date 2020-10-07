from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import pysc2

import importlib
import threading

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import run_loop
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from tf_agents.environments import utils

# based on https://github.com/deepmind/pysc2/blob/master/pysc2/bin/agent.py and https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py

class PySC2Environment(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()

        self.env = self.setup_env()

    def action_spec(self):
        return self.env.observation_spec()

    def observation_spec(self):
        return self.env.action_spec()

    def _reset(self):
        return self.env.reset()

    def _step(self, action):
        return self.env.step(action)

    def setup_env(self):

        FLAGS = flags.FLAGS
        flags.DEFINE_bool("render", True, "Whether to render with pygame.")
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
        flags.DEFINE_bool("use_raw_units", False,
                          "Whether to include raw units.")
        flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

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

        flags.DEFINE_string("map", None, "Name of a map to use.")
        flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
        flags.mark_flag_as_required("map")

        # hardcoding two players down below in the code

        agent_classes = []
        players = []

        agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
        agent_cls = getattr(importlib.import_module(agent_module), agent_name)
        agent_classes.append(agent_cls)

        players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                               FLAGS.agent_name or agent_name))

        players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                 sc2_env.Difficulty[FLAGS.difficulty],
                                 sc2_env.BotBuild[FLAGS.bot_build]))

        return sc2_env.SC2Env(map_name="Simple64",
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

def main(argv):
    # here is the test for the TF-Agents wrapper
    environment = PySC2Environment()
    utils.validate_py_environment(environment, episodes=3)

if __name__ == '__main__':
  app.run(main)
