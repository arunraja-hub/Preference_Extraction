#!/usr/bin/env python
from vizdoom import *

game = DoomGame()
game.load_config("custom.cfg")
game.set_labels_buffer_enabled(True)
game.set_episode_timeout(1000)
game.set_mode(Mode.SPECTATOR)
game.init()

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        game.advance_action()
        print(game.get_total_reward())
    r = game.get_total_reward()
    print("Reward", r)