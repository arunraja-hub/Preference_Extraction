import gin

from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
import environment_specs

@gin.configurable
class Exporter(object):
    
    def __init__(self, root_dir, env_load_fn=suite_gym.load, env_name='CartPole-v0', agent_class=None):
        
        if not agent_class:
            raise ValueError('The `agent_class` parameter of Exporter must be set.')
        
        env = env_load_fn(env_name)
        
        if isinstance(env, py_environment.PyEnvironment):
            self._tf_env = tf_py_environment.TFPyEnvironment(env)
            self._py_env = env
        else:
            self._tf_env = env
            self._py_env = None  # Can't generically convert to PyEnvironment.
        
        
        environment_specs.set_observation_spec(self._tf_env.observation_spec())
        environment_specs.set_action_spec(self._tf_env.action_spec())
        
        self._agent = agent_class(self._tf_env.time_step_spec(), self._tf_env.action_spec())
        
        print(self._agent)
