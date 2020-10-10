import os
import numpy as np

import gin
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.ppo.ppo_kl_penalty_agent import PPOKLPenaltyAgent
import environment_specs


def extract_weights_sample(agent, agent_class):
    if isinstance(agent, DqnAgent):
        return agent._q_network.layers[0].layers[0].get_weights()[0].copy()
    elif isinstance(agent, PPOKLPenaltyAgent):
        return agent.actor_net.layers[0].layers[0].get_weights()[0].copy()

def flatten_model(model_nested):
    def get_layers(layers):
        layers_flat = []
        for layer in layers:
            try:
                layers_flat.extend(get_layers(layer.layers))
            except AttributeError:
                layers_flat.append(layer)
        return layers_flat

    model_flat = tf.keras.models.Sequential(get_layers(model_nested.layers))
    
    return model_flat
    
@gin.configurable
class Exporter(object):
    
    def __init__(self, 
                 root_dir, 
                 checkpoint_no=None, 
                 env_load_fn=suite_gym.load, 
                 env_name='CartPole-v0', 
                 agent_class=None,
                 verify_restore_success=True):
        
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
        if verify_restore_success:
            pre_restore_weigths = extract_weights_sample(self._agent, agent_class)
        
        checkpoint_dir = os.path.join(root_dir, 'train', 'policy')
        checkpoint_state = tf.train.get_checkpoint_state(checkpoint_dir)
        checkpoint_paths = list(checkpoint_state.all_model_checkpoint_paths)
        if checkpoint_no is None:
            checkpoint = checkpoint_paths[-1]
        else:
            checkpoint = [x for x in checkpoint_paths if str(checkpoint_no) in x][0]
        
        policy_checkpoint = tf.train.Checkpoint(policy=self._agent.policy)
        load_status = policy_checkpoint.restore(checkpoint).expect_partial()
        load_status.initialize_or_restore()
        
        if verify_restore_success:
            post_restore_weigths = extract_weights_sample(self._agent, agent_class)
            post_equal_pre = np.allclose(pre_restore_weigths, post_restore_weigths, rtol=1e-05, atol=1e-08)
            assert post_equal_pre is False, 'Checkpoint restoring did not change weigths'
            
        if isinstance(self._agent, DqnAgent):
            model = flatten_model(self._agent._q_network) 
        elif isinstance(self._agent, PPOKLPenaltyAgent):
            model = flatten_model(self._agent.actor_net)
        
        model.build(input_shape=(1,) + env._observation_spec.shape)
        model.summary()
        model.save(os.path.join(root_dir, 'saved_model'))
