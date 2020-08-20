#!/usr/bin/env bash

rllib rollout \
--run PPO \
--use-shelve \
--no-render \
--episodes 5 \
--out /tmp/atari-brakeout.ray_rollout \
/tmp/ray-results/atari-breakout/PPO_BreakoutNoFrameskip-v4_0_2020-08-19_12-12-11eu5iymda/checkpoint_1/checkpoint-1
