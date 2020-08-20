#!/usr/bin/env bash

rld attribute \
--rllib \
--out /tmp/atari-brakeout.rld \
./config.py \
/tmp/atari-brakeout.ray_rollout
