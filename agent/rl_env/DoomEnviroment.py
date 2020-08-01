import os
from math import floor
import cv2
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from vizdoom import DoomGame
from vizdoom import GameVariable
import gin
import random
import imageio
import imageio.core.util
from collections import namedtuple

class DoomVar:
    def __init__(self, name, doom_var, reward, add_channel, max_val):
        self.name = name
        self.doom_var = doom_var
        self.reward = reward
        self.add_channel = add_channel
        self.max_val = max_val
        self.pre_val = None

def silence_imageio_warning(*args, **kwargs):
    pass

imageio.core.util._precision_warn = silence_imageio_warning

"""
    DoomEnviroment Class, code adapted from
    https://github.com/arconsis/blog-playing-doom-with-tf-agents/blob/master/doom/DoomEnvironment.py
"""

@gin.configurable
class DoomEnvironment(py_environment.PyEnvironment):

    def __init__(self, config_name, frame_skip, episode_timeout, obs_shape, start_ammo, living_reward, kill_imp_reward, kill_demon_reward, ammo_reward, health_reward):
        super().__init__()

        self.obs_shape = obs_shape
        self._game = self.configure_doom(config_name, episode_timeout)
        self._num_actions = self._game.get_available_buttons_size()
        self._frame_skip = frame_skip
        self.start_ammo = start_ammo
        self._living_reward = living_reward
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=self._num_actions - 1, name='action')
        
        all_channels = 6
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.obs_shape[0], self.obs_shape[1], all_channels), dtype=np.float32, minimum=0, maximum=1, name='observation')

        self._doom_vars = [
            DoomVar(name='kill_imp', doom_var=GameVariable.USER1, reward=kill_imp_reward, add_channel=False, max_val=None),
            DoomVar(name='kill_demon', doom_var=GameVariable.USER2, reward=kill_demon_reward, add_channel=False, max_val=None),
            DoomVar(name='health', doom_var=GameVariable.HEALTH, reward=health_reward, add_channel=True, max_val=100),
            DoomVar(name='ammo', doom_var=GameVariable.AMMO6, reward=ammo_reward, add_channel=True, max_val=self.start_ammo),
        ]
        
    @staticmethod
    def configure_doom(config_name, episode_timeout):
        game = DoomGame()
        game.load_config(config_name)
        game.set_window_visible(False)
        game.set_episode_timeout(episode_timeout)
        game.init()
        return game

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._game.new_episode()
        for _ in range(self.start_ammo):
            self._game.send_game_command("give CellPack_Single")
        self.take_action(0)

        for i in range(len(self._doom_vars)):
            self._doom_vars[i].pre_val = self._game.get_game_variable(self._doom_vars[i].doom_var)

        return ts.restart(self.get_screen_buffer_preprocessed())

    def _step(self, action):
        if self._game.is_episode_finished():
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        for i in range(self._frame_skip):
            if i == 0:
                self.take_action(action)
            else:
                self.take_action(0)

            if self._game.is_episode_finished():
                return ts.termination(self.get_screen_buffer_preprocessed(), self.get_reward())

        return ts.transition(self.get_screen_buffer_preprocessed(), self.get_reward())

    def get_reward(self):
        reward = self._living_reward * self._frame_skip
        for i in range(len(self._doom_vars)):
            current_val = self._game.get_game_variable(self._doom_vars[i].doom_var)
            reward += (current_val - self._doom_vars[i].pre_val) * self._doom_vars[i].reward
            self._doom_vars[i].pre_val = current_val

        return reward

    def take_action(self, action):
        one_hot = [0] * self._num_actions
        one_hot[action] = 1

        return self._game.make_action(one_hot)

    def render(self, mode='rgb_array'):
        """ Return image for rendering. """
        return (self.get_screen_buffer_preprocessed() * 1)[:,:,:3]


    def get_screen_buffer_preprocessed(self):
        """
            Preprocess frame for agent by:
            - cutout interesting square part of screen
            - downsample cutout to self.obs_shape (same as used for atari games)
            - normalize images to interval [0,1]
        """
        frame = self.get_screen_buffer_frame()
        frame = frame[40:-30, :]
        # The cv2 dims are backwards
        resized = cv2.resize(frame, (self.obs_shape[1], self.obs_shape[0]))
        resized = np.divide(resized, 255, dtype=np.float32)

        for doom_var in self._doom_vars:
            if doom_var.add_channel:
                val = float(self._game.get_game_variable(doom_var.doom_var)) / float(doom_var.max_val)
                resized = np.dstack((resized, self.convert_to_channel(val)))

        resized = np.dstack((resized, self.get_remaining_time_channel()))

        return resized.astype(np.float32)

    def convert_to_channel(self, value):
        # Make sure don't go below 0 cause of round error
        new_val = value / 1.2 + .1
        return np.ones(self.obs_shape) * new_val

    def get_remaining_time_channel(self):
        remaining_time = self._game.get_episode_timeout() - self._game.get_episode_time()
        return self.convert_to_channel(remaining_time / float(self._game.get_episode_timeout()))

    def get_screen_buffer_frame(self):
        """ Get current screen buffer or an empty screen buffer if episode is finished"""
        if self._game.is_episode_finished():
            return np.zeros((240, 320, 3), dtype=np.float32)
        else:
            return np.rollaxis(self._game.get_state().screen_buffer, 0, 3)

class SaveStateWrapper(wrappers.PyEnvironmentBaseWrapper):
    def __init__(self, env, path, save_prob):
        super(SaveStateWrapper, self).__init__(env)
        self.path = path
        self.save_prob = save_prob
        os.makedirs(self.path, exist_ok=True)

        self.save_num = 0

    def convert_img(self, img):
        return cv2.cvtColor(np.float32(img * 255), cv2.COLOR_RGB2BGR)

    def _step(self, action):
        time_step = self._env.step(action)

        if random.random() < self.save_prob:
            cv2.imwrite(os.path.join(self.path, str(self.save_num)+'_0.png'), self.convert_img(time_step.observation[:, :, :3]))

            for i in range(3, time_step.observation.shape[2]):
                cv2.imwrite(os.path.join(self.path, str(self.save_num) + '_' + str(i) + '.png'),
                            self.convert_img(time_step.observation[:, :, i]))

            self.save_num += 1

        return time_step
    
class SaveVideoWrapper(wrappers.PyEnvironmentBaseWrapper):
    def __init__(self, env, filename):
        super(SaveVideoWrapper, self).__init__(env)
        self.filename = filename
        self.video = imageio.get_writer(filename, fps=30)

    def _step(self, action):
        time_step = self._env.step(action)
        self.video.append_data(self._env.render())
        return time_step

@gin.configurable
def tf_agents_env(_):
    return DoomEnvironment()

@gin.configurable
def tf_agents_env_with_video(_):
    return SaveVideoWrapper(tf_agents_env(None), 'states_video.mp4')
