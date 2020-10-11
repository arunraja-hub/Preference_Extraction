import os
import numpy as np
import collections
import pickle

import gin
import tensorflow as tf
from tensorflow.python.lib.io import file_io
        
from tf_agents.environments import py_environment
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.agents.ppo.ppo_kl_penalty_agent import PPOKLPenaltyAgent
from tqdm import tqdm
        
from rl_env.DoomEnvironment import DoomEnvironment
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


class Trajectory(
    collections.namedtuple('Trajectory', [
        'step_type',
        'observation',
        'action',
        'policy_info',
        'next_step_type',
        'reward',
        'discount',
    ])):
  """Stores the observation the agent saw, the action it took
  and preference labels, if specified"""
  __slots__ = ()

@gin.configurable
class Exporter(object):
    
    def __init__(self, 
                 root_dir, 
                 checkpoint_no=None, 
                 env_load_fn=suite_gym.load, 
                 env_name='CartPole-v0', 
                 agent_class=None,
                 collect_data=0,
                 verify_restore_success=True):
        
        if not agent_class:
            raise ValueError('The `agent_class` parameter of Exporter must be set.')
        
        self.env = env_load_fn(env_name)
        
        if isinstance(self.env, py_environment.PyEnvironment):
            self._tf_env = tf_py_environment.TFPyEnvironment(self.env)
            self._py_env = self.env
        else:
            self._tf_env = self.env
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
            checkpoint_no = 'last'
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
        
        input_shape = self.env._observation_spec.shape
        model.build(input_shape= (1,) + input_shape)
        model._set_inputs(tf.keras.Input(shape=input_shape))
        model.summary()
        model_dir = os.path.join(root_dir, f'saved_model_cp_{checkpoint_no}')
        model.save(model_dir)
        
        if collect_data > 0:
            self.generate_experience_data(steps=collect_data, save_dir=root_dir)
   
    def generate_experience_data(self, steps, save_dir):
        
        time_step = self._tf_env.reset()
        observations = []
        actions = []
        labels = []
        
        for _ in tqdm(range(steps), 'Generating experience data'):
            action = self._agent.policy.action(time_step).action
            time_step = self._tf_env.step(action=action)
            
            label = {}
            if isinstance(self.env._env, DoomEnvironment):            
                state = self._tf_env.envs[0]._game.get_state()
                self._tf_env.envs[0]._game.advance_action()
                if state is not None:
                    deamons = [lbl for lbl in state.labels if lbl.object_name == 'Demon']
                    if len(deamons) > 0:
                        label['object_angle'] = int(deamons[0].object_angle)
                        label['distance_from_wall'] = abs(deamons[0].object_position_x)

            observations.append(time_step.observation)
            actions.append(action.numpy()[0])
            labels.append(label)
            
        observations = np.array([ob.numpy()[0] for ob in observations])
        actions = np.array(actions)
        labels = np.array(labels)

        exp_data = Trajectory(
            observation=observations,
            action=actions,
            policy_info={'satisfaction': labels},
            step_type=(),
            next_step_type=(),
            reward=(),
            discount=())
        
        file_path = os.path.join(save_dir, f'exp_data_{steps}.pkl')
        with file_io.FileIO(file_path, mode='wb') as f:
            pickle.dump([exp_data], f)
        