import numpy as np
from tf_agents.environments import py_environment
from tf_agents.environments import wrappers
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import pysc2

class PySC2Environment(py_environment.PyEnvironment):
    def __init__(self):
        super().__init__()
        pass

    def action_spec(self):
        #return self._action_spec
        pass

    def observation_spec(self):
        #return self._observation_spec
        pass

    def _reset(self):
        pass

    def _step(self, action):
        pass
