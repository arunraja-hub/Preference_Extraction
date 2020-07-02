#!/usr/bin/env python

from vizdoom import *
import random
import time

game = DoomGame()
game.load_config("custom.cfg")
game.set_labels_buffer_enabled(True)
game.set_mode(Mode.SPECTATOR)
game.init()

episodes = 10
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        game.advance_action()
        labels = state.labels

        human_label = None
        for label in labels:
            if label.object_name == 'Demon':
                human_label = label
            if label.object_name == 'DoomPlayer':
                player_label = label
        # if human_label is not None:
        #     print(human_label.object_angle)
        # else:
        #     print(None)
        # print("player_label", player_label.object_angle)

        if human_label is None:
            print("Dead")
        else:
            if human_label.object_angle < 90 or human_label.object_angle > 270:
                print("Attacking monster")
            else:
                print("Attacking agent")
    time.sleep(2)