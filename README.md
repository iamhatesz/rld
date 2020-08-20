# rld

![Build and test](https://github.com/iamhatesz/rld/workflows/Build%20and%20test/badge.svg)

A development tool for evaluation and interpretability of reinforcement learning agents.

![rld demo gif](https://imgur.com/hodTIcj.gif)

## Installation

```bash
pip install rld
```

## Usage

Firstly, calculate attributations for your rollout using:

```bash
rld attribute [--rllib] [--out <ROLLOUT>] config.py <INPUT_ROLLOUT>
```

This will take `INPUT_ROLLOUT` (possibly in the Ray RLlib format, if `--rllib` is set)
and calculate attributations for each timestep in each trajectory,
using the configuration stored in `config.py`.
The output file will be stored as `ROLLOUT`.
See the `Config` class for possible configuration.

Once the attributations are calculated, you can visualize them using:

```bash
rld start --viewer <VIEWER_ID> <ROLLOUT>
```

## Description

rld provides a set of tools to evaluate and understand behaviors of reinforcement
learning agents. Under the hood, rld uses [Captum](https://captum.ai/) to calculate
attributations of observation components. rld is also integrated with
[Ray RLlib](https://ray.io/) library and allows to load agents trained in RLlib.

### Current limitations

rld is currently in its early development stage, thus the functionality is very limited.

#### RL algorithms

rld is algorithm-agnostic, but currently it is more suitable for policy-based methods.
This is due to the fact that the `Model` is now expected to output logits for a given
observation. This, however, will change in the future, and rld will support more
algorithms.

#### Viewers

This is the list of viewers, which ship with rld:
* `none`
* `cartpole`
* `atari`

You can easily create your own viewer, for your own environment, but to make it visible
for rld, you have to rebuild the project. This will be improved in the future.

#### Observation and action spaces

The table below presents currently supported observation and action spaces.

<table>
    <tr>
        <td></td>
        <td></td>
        <td colspan="2"><strong>Action space</strong></td>
    </tr>
    <tr>
        <td></td>
        <td></td>
        <td>Discrete</td>
        <td>MultiDiscrete</td>
    </tr>
    <tr>
        <td rowspan="3"><strong>Obs space</strong></td>
        <td>Box</td>
        <td>:heavy_check_mark:</td>
        <td>:heavy_check_mark:</td>
    </tr>
    <tr>
        <td>Dict</td>
        <td>:heavy_check_mark:</td>
        <td>:heavy_check_mark:</td>
    </tr>
</table>

## Roadmap

See the [issues](https://github.com/iamhatesz/rld/issues) page to see the list of
features planned for the future releases. If you have your own ideas,
you are encouraged to post them there.
