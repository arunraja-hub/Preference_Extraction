"""Allows playing the environment from the tf agents environment. That way it includes all post processing"""

import gin
from  rl_env.DoomEnviroment import tf_agents_env
import gin.tf.external_configurables
import pygame
from pygame.locals import *


def get_action():
    pygame.event.clear()
    while True:
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == KEYDOWN and event.key == K_d:
            return 1
        if event.type == KEYDOWN and event.key == K_a:
            return 2
        if event.type == KEYDOWN and event.key == K_s:
            return 0
        if event.type == KEYDOWN and event.key == K_SPACE:
            return 3


gin.parse_config_files_and_bindings(['configs/dqn.gin'], '', skip_unknown=True)
env = tf_agents_env(None)

pygame.init()
size = (600, 1000)
display = pygame.display.set_mode((size[1], size[0]))

total_reward = 0
time_step = env.reset()
while not time_step.is_last():
    obs = time_step.observation[:, :, :3]

    print("health", time_step.observation[0, 0, 3])
    print("time", time_step.observation[0, 0, 4])
    print("ammo", time_step.observation[0, 0, 5])

    obs *= 255

    surf = pygame.surfarray.make_surface(obs)
    surf = pygame.transform.scale(surf, size)
    surf = pygame.transform.rotate(surf, -90)

    display.blit(surf, (0, 0))
    pygame.display.update()

    action = get_action()

    time_step = env.step(action)
    total_reward += time_step.reward

    print("total_reward", total_reward)

pygame.quit()
