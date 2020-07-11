import gin
from tf_agents.environments import utils
from  rl_env.DoomEnviroment import tf_agents_env, SaveStateWrapper
import gin.tf.external_configurables


if __name__ == "__main__":
    gin.parse_config_files_and_bindings(['configs/dqn.gin'], '', skip_unknown=True)

    environment = SaveStateWrapper(tf_agents_env(None), 'saved_env_states', 0.1)

    utils.validate_py_environment(environment, episodes=5)