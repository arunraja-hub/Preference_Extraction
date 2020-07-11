import os
from math import floor
import cv2
import numpy as np
from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from vizdoom import DoomGame
from vizdoom import GameVariable
import matplotlib
import gin
import random

"""
    DoomEnviroment Class, code adapted from
    https://github.com/arconsis/blog-playing-doom-with-tf-agents/blob/master/doom/DoomEnvironment.py
"""

@gin.configurable
class DoomEnvironment(py_environment.PyEnvironment):

    def __init__(self, config_name, episode_timeout=1000, obs_shape = (60, 100), timeout_channel=True, ammo_channel=True):
        super().__init__()

        self.obs_shape = obs_shape

        self._game = self.configure_doom(config_name, episode_timeout, timeout_channel)
        self.timeout_channel = timeout_channel
        self.ammo_channel = ammo_channel
        self._num_actions = self._game.get_available_buttons_size()
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64, minimum=0, maximum=self._num_actions - 1, name='action')
        
        all_channels = 5
        if not timeout_channel:
            all_channels -= 1
        if not ammo_channel:
            all_channels -= 1
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
        reward = self._game.make_action(one_hot)

        # return transition depending on game state
        if self._game.is_episode_finished():
            return ts.termination(self.get_screen_buffer_preprocessed(), reward)
        else:
            return ts.transition(self.get_screen_buffer_preprocessed(), reward)

    def render(self, mode='rgb_array'):
        """ Return image for rendering. """
        return (self.get_screen_buffer_preprocessed() * 255)[:,:,:3]


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
        if self.timeout_channel:
            resized = np.dstack((resized, self.get_remaining_time_channel()))
        if self.ammo_channel:
            resized = np.dstack((resized, self.get_remaining_ammo_channel()))
        return np.divide(resized, 255, dtype=np.float32)

    def get_remaining_time_channel(self):
        time_channel = np.zeros(self.obs_shape[0] * self.obs_shape[1])
        time_left = self._game.get_episode_timeout() - self._game.get_episode_time()
        time_channel[:time_left * 10] = 1
        return time_channel.reshape(self.obs_shape)

    def get_remaining_ammo_channel(self):
        ammo_channel = np.zeros(self.obs_shape[0] * self.obs_shape[1])
        ammo_left = self.get_weapon_remaining_ammo()
        ammo_channel[:ammo_left * 10] = 1
        return ammo_channel.reshape(self.obs_shape)

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
        return (img * 255).astype(int)

    def _step(self, action):
        time_step = self._env.step(action)

        if random.random() < self.save_prob:
            cv2.imwrite(os.path.join(self.path, str(self.save_num)+'_0.png'), self.convert_img(time_step.observation[:, :, :3]))

            for i in range(3, time_step.observation.shape[2]):
                cv2.imwrite(os.path.join(self.path, str(self.save_num) + '_' + str(i) + '.png'),
                            self.convert_img(time_step.observation[:, :, i]))

            self.save_num += 1

        return time_step



@gin.configurable
def tf_agents_env(_):
    return DoomEnvironment()

if __name__ == "__main__":
    environment = SaveStateWrapper(DoomEnvironment('custom.cfg'), 'saved_env_states', .01)


    utils.validate_py_environment(environment, episodes=5)
