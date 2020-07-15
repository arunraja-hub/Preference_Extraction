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

"""
    DoomEnviroment Class, code adapted from
    https://github.com/arconsis/blog-playing-doom-with-tf-agents/blob/master/doom/DoomEnvironment.py
"""

@gin.configurable
class DoomEnvironment(py_environment.PyEnvironment):

    def __init__(self, config_name, episode_timeout=1000, obs_shape = (60, 100), frame_skip=4, timeout_channel=True, ammo_channel=True):
        super().__init__()

        self.obs_shape = obs_shape
        self._game = self.configure_doom(config_name, episode_timeout, timeout_channel)
        self.timeout_channel = timeout_channel
        self.ammo_channel = ammo_channel
        self._num_actions = self._game.get_available_buttons_size()
        self._frame_skip = frame_skip
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=self._num_actions - 1, name='action')
        
        all_channels = 4 + timeout_channel + ammo_channel
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self.obs_shape[0], self.obs_shape[1], all_channels), dtype=np.float32, minimum=0, maximum=1, name='observation')
        
    @staticmethod
    def configure_doom(config_name, episode_timeout, timeout_channel):
        game = DoomGame()
        game.load_config(config_name)
        game.set_window_visible(False)
        if timeout_channel:
            game.set_episode_timeout(episode_timeout)
        game.init()
        return game

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._game.new_episode()
        return ts.restart(self.get_screen_buffer_preprocessed())

    def _step(self, action):
        if self._game.is_episode_finished():
            # The last action ended the episode. Ignore the current action and start a new episode.
            return self.reset()

        # construct one hot encoded action as required by ViZDoom
        one_hot = [0] * self._num_actions
        one_hot[action] = 1

        # execute action and receive reward
        reward = self._game.make_action(one_hot, self._frame_skip)

        # return transition depending on game state
        if self._game.is_episode_finished():
            return ts.termination(self.get_screen_buffer_preprocessed(), reward)
        else:
            return ts.transition(self.get_screen_buffer_preprocessed(), reward)

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
        resized = np.dstack((resized, self.get_health_channel()))  # Health channel is always index 3
        if self.timeout_channel:
            resized = np.dstack((resized, self.get_remaining_time_channel()))
        if self.ammo_channel:
            resized = np.dstack((resized, self.get_remaining_ammo_channel()))
        return resized.astype(np.float32)

    def convert_to_channel(self, value):
        # Make sure don't go below 0 cause of round error
        new_val = value / 1.2 + .1
        return np.ones(self.obs_shape) * new_val

    def get_remaining_time_channel(self):
        remaining_time = self._game.get_episode_timeout() - self._game.get_episode_time()
        return self.convert_to_channel(remaining_time / float(self._game.get_episode_timeout()))

    def get_remaining_ammo_channel(self):
        remaining_ammo = self.get_weapon_remaining_ammo()
        return self.convert_to_channel(remaining_ammo / 50.0)

    def get_health_channel(self):
        remaining_health = self._game.get_game_variable(GameVariable.HEALTH)
        return self.convert_to_channel(remaining_health / 100.0)

    def get_weapon_remaining_ammo(self):
        for am_ix, ammo in enumerate([GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3,
                              GameVariable.AMMO4, GameVariable.AMMO5, GameVariable.AMMO6,
                              GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9]):
            if (am_ix+1) == int(self._game.get_game_variable(GameVariable.SELECTED_WEAPON)):
                return int(self._game.get_game_variable(ammo))
        return 0

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
        for frame in range(self._env._frame_skip):
            self.video.append_data(self._env.render())
        return time_step

@gin.configurable
def tf_agents_env(_):
    return DoomEnvironment()
