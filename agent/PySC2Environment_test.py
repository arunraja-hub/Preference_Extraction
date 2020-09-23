#import gin
from tf_agents.environments import utils
#from  rl_env.DoomEnviroment import tf_agents_env, SaveStateWrapper, SaveVideoWrapper
#import gin.tf.external_configurables
from rl_env.PySC2Environment import PySC2Environment


if __name__ == "__main__":
    #gin.parse_config_files_and_bindings(['configs/dqn.gin'], '', skip_unknown=True)
    environment = PySC2Environment()
    # environment = SaveVideoWrapper(tf_agents_env(None), 'states_video.mp4')
    utils.validate_py_environment(environment, episodes=3)
