import gin
from tf_agents.environments import utils
from  rl_env.DoomEnviroment import tf_agents_env, SaveStateWrapper, SaveVideoWrapper
import gin.tf.external_configurables


if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['configs/dqn.gin'], '', skip_unknown=True)
    environment = SaveStateWrapper(tf_agents_env(None), 'saved_env_states', 0.1)
    environment = SaveVideoWrapper(tf_agents_env(None), 'states_video.mp4')
    utils.validate_py_environment(environment, episodes=3)