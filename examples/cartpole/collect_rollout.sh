#!/usr/bin/env bash

rllib rollout \
--run PPO \
--use-shelve \
--no-render \
--episodes 5 \
--out /tmp/cartpole.ray_rollout \
/tmp/ray-results/cartpole/PPO_CartPole-v1_0_2020-08-17_17-07-12s_bybqva/checkpoint_17/checkpoint-17
