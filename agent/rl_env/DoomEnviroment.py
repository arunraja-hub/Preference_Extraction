import os
from math import floor
import cv2
import numpy as np
from tf_agents.environments import py_environment, utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from vizdoom import DoomGame
from vizdoom import GameVariable
import gin

"""
    DoomEnviroment Class, code adapted from
    https://github.com/arconsis/blog-playing-doom-with-tf-agents/blob/master/doom/DoomEnvironment.py
"""

@gin.configurable
class DoomEnvironment(py_environment.PyEnvironment):

    def __init__(self, config_name, episode_timeout=1000, timeout_channel=True, ammo_channel=True):
        super().__init__()

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
            shape=(84, 84, all_channels), dtype=np.float32, minimum=0, maximum=1, name='observation')
        
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
            - downsample cutout to 84x84 (same as used for atari games)
            - normalize images to interval [0,1]
        """
        frame = self.get_screen_buffer_frame()
        cutout = frame[10:-10, 30:-30]
        resized = cv2.resize(cutout, (84, 84))
        if self.timeout_channel:
            resized = np.dstack((resized, self.get_remaining_time_channel()))
        if self.ammo_channel:
            resized = np.dstack((resized, self.get_remaining_ammo_channel()))
        return np.divide(resized, 255, dtype=np.float32)

    def get_remaining_time_channel(self):
        time_channel = np.zeros((84 * 84))
        time_left = self._game.get_episode_timeout() - self._game.get_episode_time()
        time_channel[:time_left * 10] = 1
        return time_channel.reshape((84, 84))

    def get_remaining_ammo_channel(self):
        ammo_channel = np.zeros((84 * 84))
        ammo_left = self.get_weapon_remaining_ammo()
        ammo_channel[:ammo_left * 10] = 1
        return ammo_channel.reshape((84, 84))

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

@gin.configurable
def tf_agents_env(_):
    return DoomEnvironment()

if __name__ == "__main__":
	environment = DoomEnvironment('custom.cfg')
	utils.validate_py_environment(environment, episodes=5)
