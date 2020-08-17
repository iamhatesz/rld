#!/usr/bin/env bash

rld attribute \
--rllib \
--out /tmp/cartpole.rld \
./config.py \
/tmp/cartpole.ray_rollout
