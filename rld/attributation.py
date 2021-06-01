import operator
import sys
from collections import abc
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import reduce
from itertools import accumulate, product
from typing import Optional, Union, List, Tuple

import gym
import numpy as np
import torch
from captum.attr import IntegratedGradients
from gym.spaces import flatten, unflatten

from rld.exception import ActionSpaceNotSupported, EnumValueNotFound
from rld.model import Model, RecurrentModel, pack_array, unpack_array
from rld.rollout import (
    Trajectory,
    Timestep,
    DiscreteActionAttributation,
    MultiDiscreteActionAttributation,
    remove_channel_dim_from_image_space,
    Attributation,
)
from rld.typing import BaselineBuilder, AttributationLike, ActionLike, HiddenState


class AttributationNormalizationMode(IntEnum):
    ALL = 0
    POSITIVE = 1
    NEGATIVE = 2
    ABSOLUTE_VALUE = 3


class AttributationNormalizer:
    """
    Based on Captum image attributation visualization technique.
    """

    def __init__(
        self,
        obs_space: gym.Space,
        obs_image_channel_dim: Optional[int],
        mode: AttributationNormalizationMode,
        outlier_percentile: Union[int, float],
    ):
        self.obs_space = obs_space
        self.obs_image_channel_dim = obs_image_channel_dim
        self.mode = mode
        self.outlier_percentile = outlier_percentile

    def transform(self, attr: AttributationLike) -> AttributationLike:
        obs_space = self.obs_space
        if self.obs_image_channel_dim is not None:
            attr = np.sum(attr, axis=self.obs_image_channel_dim)
            obs_space = remove_channel_dim_from_image_space(obs_space)
        attr = flatten(self.obs_space, attr)
        if self.mode == AttributationNormalizationMode.ALL:
            scaling_factor = self._calculate_safe_scaling_factor(np.abs(attr))
        elif self.mode == AttributationNormalizationMode.POSITIVE:
            attr = (attr > 0) * attr
            scaling_factor = self._calculate_safe_scaling_factor(attr)
        elif self.mode == AttributationNormalizationMode.NEGATIVE:
            attr = (attr < 0) * attr
            scaling_factor = -self._calculate_safe_scaling_factor(np.abs(attr))
        elif self.mode == AttributationNormalizationMode.ABSOLUTE_VALUE:
            attr = np.abs(attr)
            scaling_factor = self._calculate_safe_scaling_factor(attr)
        else:
            raise EnumValueNotFound(self.mode, AttributationNormalizationMode)
        attr_norm = self._scale(attr, scaling_factor)
        return unflatten(obs_space, attr_norm)

    def _calculate_safe_scaling_factor(self, attr: AttributationLike) -> float:
        sorted_vals = np.sort(attr.flatten())
        cum_sums = np.cumsum(sorted_vals)
        threshold_id = np.where(
            cum_sums >= cum_sums[-1] * 0.01 * (100 - self.outlier_percentile)
        )[0][0]
        return sorted_vals[threshold_id]

    @staticmethod
    def _scale(attr: AttributationLike, scaling_factor: float) -> AttributationLike:
        if abs(scaling_factor) < 1e-5:
            return np.clip(attr, -1, 1)
        attr_norm = attr / scaling_factor
        return np.clip(attr_norm, -1, 1)


class AttributationTarget(IntEnum):
    # The target equals to the actions picked by an agent
    PICKED = 0
    # The target chosen as argmax from action distribution
    HIGHEST = 1
    # 3 highest targets (calculated in the same way as `HIGHEST`)
    TOP3 = 3
    # 5 highest targets (calculated in the same way as `HIGHEST`)
    TOP5 = 5
    # All available targets
    ALL = sys.maxsize


@dataclass
class TimestepAttributationBatch:
    inputs: torch.Tensor
    state: Optional[HiddenState]
    baselines: torch.Tensor
    targets: torch.Tensor
    actions: List[ActionLike]
    action_probs: List[float]
    probs: torch.Tensor
    timestep: Timestep


class AttributationTrajectoryIterator(abc.Iterator):
    def __init__(
        self,
        trajectory: Trajectory,
        model: Model,
        baseline: Optional[BaselineBuilder] = None,
        target: AttributationTarget = AttributationTarget.PICKED,
    ):
        self.trajectory = trajectory
        self.model = model
        self.baseline = baseline
        self.target = target
        self._it = iter(self.trajectory)
        self._model_recurrent = isinstance(self.model, RecurrentModel)
        self._state: Optional[
            HiddenState
        ] = self.model.initial_state() if self._model_recurrent else None

    @torch.no_grad()
    def __next__(self) -> TimestepAttributationBatch:
        try:
            timestep = next(self._it)
        except StopIteration:
            raise StopIteration

        # inputs = self.model.flatten_obs(timestep.obs)
        inputs = pack_array(timestep.obs, self.model.obs_space())

        if self.baseline is not None:
            baselines = self.baseline(inputs)
        else:
            baselines = np.zeros_like(inputs)

        inputs = torch.tensor(inputs, device=self.model.input_device()).unsqueeze(dim=0)
        baselines = torch.tensor(
            baselines, device=self.model.output_device()
        ).unsqueeze(dim=0)

        if self._model_recurrent:
            out = self.model(inputs, self._state)
            state = self.model.last_output_state().to(device=self.model.input_device())
            self._state = state
            logits = out.squeeze(dim=0).to(device=self.model.input_device())
        else:
            logits = (
                self.model(inputs).squeeze(dim=0).to(device=self.model.input_device())
            )

        if isinstance(self.model.action_space(), gym.spaces.Discrete):
            probs = torch.softmax(logits, dim=-1)
        elif isinstance(self.model.action_space(), gym.spaces.MultiDiscrete):
            probs = _multicategorical_softmax(
                logits, list(self.model.action_space().nvec)
            )
        elif isinstance(self.model.action_space(), gym.spaces.Tuple):
            probs = _multicategorical_softmax(
                logits, n_vecs(self.model.action_space())
            )
        else:
            raise ActionSpaceNotSupported(self.model.action_space())

        targets = []

        picked_action = timestep.action
        raw_picked_action = _action_to_raw_action(
            self.model.action_space(), timestep.action
        )

        if isinstance(self.model.action_space(), gym.spaces.Discrete):
            prob_of_picked_action = probs[picked_action].item()
        elif isinstance(self.model.action_space(), gym.spaces.MultiDiscrete):
            prob_of_picked_action = (
                torch.stack([probs[s] for s in raw_picked_action]).prod().item()
            )
        elif isinstance(self.model.action_space(), gym.spaces.Tuple):
            if len(raw_picked_action) > 1:
                prob_of_picked_action = (
                    torch.stack([probs[s] for s in raw_picked_action]).prod().item()
                )
            else:
                prob_of_picked_action = probs[picked_action].item()
        else:
            raise ActionSpaceNotSupported(self.model.action_space())

        # Always add the picked action to the targets list
        targets.append((picked_action, raw_picked_action, prob_of_picked_action))

        # Calculate additional targets list
        if self.target in (
            AttributationTarget.TOP3,
            AttributationTarget.TOP5,
            AttributationTarget.ALL,
        ):
            if isinstance(self.model.action_space(), gym.spaces.Discrete):
                # Get total available actions
                num_actions = logits.numel()
                num_top = min(int(self.target.value), num_actions)
                top_probs_with_index = probs.topk(k=num_top, dim=-1)
                for index, prob in zip(
                    top_probs_with_index.indices, top_probs_with_index.values
                ):
                    action = index.item()
                    raw_action = _action_to_raw_action(
                        self.model.action_space(), action
                    )
                    prob = prob.item()
                    targets.append((action, raw_action, prob))
            elif isinstance(self.model.action_space(), gym.spaces.MultiDiscrete):
                # Extract probs for each sub-action
                subs_probs = _extract_multi_discrete_action_probs(
                    self.model.action_space(), probs
                )
                # Calculate most probably actions
                top_probs_with_index = _sort_multi_discrete_action_probs(subs_probs)

                num_actions = _total_multi_discrete_actions(self.model.action_space())
                num_top = min(int(self.target.value), num_actions)
                for action, prob in top_probs_with_index[:num_top]:
                    raw_action = _action_to_raw_action(
                        self.model.action_space(), action
                    )
                    targets.append((action, raw_action, prob))
            elif isinstance(self.model.action_space(), gym.spaces.Tuple):
                # Extract probs for each sub-action
                subs_probs = _extract_tuple_discrete_action_probs(
                    self.model.action_space(), probs
                )
                # Calculate most probably actions
                top_probs_with_index = _sort_multi_discrete_action_probs(subs_probs)

                num_actions = _total_tuple_discrete_actions(self.model.action_space())
                num_top = min(int(self.target.value), num_actions)
                for action, prob in top_probs_with_index[:num_top]:
                    raw_action = _action_to_raw_action(
                        self.model.action_space(), action
                    )
                    targets.append((action, raw_action, prob))

            else:
                raise ActionSpaceNotSupported(self.model.action_space())

        inputs_all = []
        if self._model_recurrent:
            states_all = []
        baselines_all = []
        targets_all = []

        for _, target, _ in targets:
            if isinstance(self.model.action_space(), gym.spaces.Discrete):
                inputs_for_target = inputs
                if self._model_recurrent:
                    states_for_target = state
                baselines_for_target = baselines
                targets_for_target = torch.tensor(
                    target, device=self.model.output_device(), dtype=torch.long
                ).unsqueeze(dim=0)
            elif isinstance(self.model.action_space(), gym.spaces.MultiDiscrete):
                num_sub_actions = _multi_discrete_actions_count(
                    self.model.action_space()
                )
                inputs_for_target = _extend_batch_dim(inputs, num_sub_actions)
                if self._model_recurrent:
                    states_for_target = _extend_batch_dim(state, num_sub_actions)
                baselines_for_target = _extend_batch_dim(baselines, num_sub_actions)
                targets_for_target = torch.tensor(
                    target, device=self.model.output_device(), dtype=torch.long
                )
            elif isinstance(self.model.action_space(), gym.spaces.Tuple):
                num_sub_actions = _tuple_discrete_actions_count(
                    self.model.action_space()
                )
                inputs_for_target = _extend_batch_dim(inputs, num_sub_actions)
                if self._model_recurrent:
                    states_for_target = _extend_batch_dim(state, num_sub_actions)
                baselines_for_target = _extend_batch_dim(baselines, num_sub_actions)
                targets_for_target = torch.tensor(
                    target, device=self.model.output_device(), dtype=torch.long
                )
            else:
                raise ActionSpaceNotSupported(self.model.action_space())
            inputs_all.append(inputs_for_target)
            if self._model_recurrent:
                states_all.append(states_for_target)
            baselines_all.append(baselines_for_target)
            targets_all.append(targets_for_target)

        inputs_batch = torch.cat(inputs_all, dim=0)
        if self._model_recurrent:
            states_batch = torch.cat(states_all, dim=0)
        else:
            states_batch = None
        baselines_batch = torch.cat(baselines_all, dim=0)
        targets_batch = torch.cat(targets_all, dim=0)
        actions = [action for action, _, _ in targets]
        action_probs = [prob for _, _, prob in targets]

        return TimestepAttributationBatch(
            inputs=inputs_batch,
            state=states_batch,
            baselines=baselines_batch,
            targets=targets_batch,
            actions=actions,
            action_probs=action_probs,
            probs=probs,
            timestep=timestep,
        )


# TODO attribute_rollout method


def attribute_trajectory(
    trajectory_it: AttributationTrajectoryIterator,
    model: Model,
    normalizer: AttributationNormalizer,
) -> Trajectory:
    obs_space = model.obs_space()
    is_recurrent = isinstance(model, RecurrentModel)
    algo = IntegratedGradients(model)
    timesteps = []
    for batch in trajectory_it:
        if not is_recurrent:
            raw_attributation = algo.attribute(
                batch.inputs, baselines=batch.baselines, target=batch.targets
            )
        else:
            raw_attributation = algo.attribute(
                (batch.inputs, batch.state),
                baselines=(batch.baselines, batch.state),
                target=batch.targets,
            )
            raw_attributation = raw_attributation[0]
        raw_attributation = raw_attributation.detach().cpu().numpy()

        if isinstance(model.action_space(), gym.spaces.Discrete):
            attributation = Attributation(
                picked=DiscreteActionAttributation(
                    action=batch.actions[0],
                    prob=batch.action_probs[0],
                    raw=unpack_array(raw_attributation[0], obs_space),
                    normalized=normalizer.transform(
                        unpack_array(raw_attributation[0], obs_space)
                    ),
                ),
                top=[
                    DiscreteActionAttributation(
                        action=batch.actions[i],
                        prob=batch.action_probs[i],
                        raw=unpack_array(raw_attributation[i], obs_space),
                        normalized=normalizer.transform(
                            unpack_array(raw_attributation[i], obs_space)
                        ),
                    )
                    for i in range(1, len(batch.actions))
                ],
            )
        elif isinstance(model.action_space(), gym.spaces.MultiDiscrete):
            num_sub_actions = _multi_discrete_actions_count(model.action_space())
            grouped_attributation = [
                raw_attributation[i: i + num_sub_actions]
                for i in range(0, raw_attributation.shape[0], num_sub_actions)
            ]

            attributation = Attributation(
                picked=MultiDiscreteActionAttributation(
                    action=batch.actions[0],
                    prob=batch.action_probs[0],
                    raw=[
                        unpack_array(grouped_attributation[0][j], obs_space)
                        for j in range(num_sub_actions)
                    ],
                    normalized=[
                        normalizer.transform(
                            unpack_array(grouped_attributation[0][j], obs_space)
                        )
                        for j in range(num_sub_actions)
                    ],
                ),
                top=[
                    MultiDiscreteActionAttributation(
                        action=batch.actions[i],
                        prob=batch.action_probs[i],
                        raw=[
                            unpack_array(grouped_attributation[i][j], obs_space)
                            for j in range(num_sub_actions)
                        ],
                        normalized=[
                            normalizer.transform(
                                unpack_array(grouped_attributation[i][j], obs_space)
                            )
                            for j in range(num_sub_actions)
                        ],
                    )
                    for i in range(1, len(batch.actions))
                ],
            )
        elif isinstance(model.action_space(), gym.spaces.Tuple):
            num_sub_actions = _tuple_discrete_actions_count(model.action_space())
            grouped_attributation = [
                raw_attributation[i: i + num_sub_actions]
                for i in range(0, raw_attributation.shape[0], num_sub_actions)
            ]

            attributation = Attributation(
                picked=MultiDiscreteActionAttributation(
                    action=batch.actions[0],
                    prob=batch.action_probs[0],
                    raw=[
                        unpack_array(grouped_attributation[0][j], obs_space)
                        for j in range(num_sub_actions)
                    ],
                    normalized=[
                        normalizer.transform(
                            unpack_array(grouped_attributation[0][j], obs_space)
                        )
                        for j in range(num_sub_actions)
                    ],
                ),
                top=[
                    MultiDiscreteActionAttributation(
                        action=batch.actions[i],
                        prob=batch.action_probs[i],
                        raw=[
                            unpack_array(grouped_attributation[i][j], obs_space)
                            for j in range(num_sub_actions)
                        ],
                        normalized=[
                            normalizer.transform(
                                unpack_array(grouped_attributation[i][j], obs_space)
                            )
                            for j in range(num_sub_actions)
                        ],
                    )
                    for i in range(1, len(batch.actions))
                ],
            )
        else:
            raise ActionSpaceNotSupported(model.action_space())

        timesteps.append(replace(batch.timestep, attributations=attributation))

    return Trajectory(timesteps)


def _extend_batch_dim(t: torch.Tensor, new_batch_dim: int) -> torch.Tensor:
    """
    Given a tensor `t` of shape [B x D1 x D2 x ...] we output the same tensor repeated
    along the batch dimension ([new_batch_dim x D1 x D2 x ...]).
    """
    num_non_batch_dims = len(t.shape[1:])
    repeat_shape = (new_batch_dim, *(1 for _ in range(num_non_batch_dims)))
    return t.repeat(repeat_shape)


def _multi_discrete_actions_count(action_space: gym.spaces.MultiDiscrete) -> int:
    return len(action_space.nvec)


def _tuple_discrete_actions_count(action_space: gym.spaces.MultiDiscrete) -> int:
    return len(n_vecs(action_space))


def _total_multi_discrete_actions(action_space: gym.spaces.MultiDiscrete) -> int:
    return reduce(operator.mul, action_space.nvec)


def _total_tuple_discrete_actions(action_space: gym.spaces.MultiDiscrete) -> int:
    return reduce(operator.mul, n_vecs(action_space))


def _extract_multi_discrete_action_probs(
    action_space: gym.spaces.MultiDiscrete, probs: torch.Tensor
) -> List[torch.Tensor]:
    # Extract sizes of each dimension (e.g. [4, 2, 3])
    subs = action_space.nvec
    # Accumulate sizes to calculate offsets (e.g. [0, 4, 6, 9])
    # TODO Use initial=0 when moved to Python 3.8
    offsets = [0] + list(accumulate(subs, operator.add))[:-1]
    return [probs[o: o + s] for o, s in zip(offsets, subs)]


def _extract_tuple_discrete_action_probs(action_space: gym.spaces.MultiDiscrete,
                                         probs: torch.Tensor) -> List[torch.Tensor]:
    subs = n_vecs(action_space)
    offsets = [0] + list(accumulate(subs, operator.add))[:-1]
    return [probs[o: o + s] for o, s in zip(offsets, subs)]


def _sort_multi_discrete_action_probs(
    probs: List[torch.Tensor],
) -> List[Tuple[ActionLike, float]]:
    indices = [range(s.numel()) for s in probs]
    indices_tuples = [np.array(i) for i in product(*indices)]
    probs_tuples = [torch.stack(s, dim=0).prod().item() for s in product(*probs)]
    tuples = sorted(zip(indices_tuples, probs_tuples), key=lambda t: t[1], reverse=True)
    return list(tuples)


def _action_to_raw_action(action_space: gym.Space, action: ActionLike) -> ActionLike:
    if isinstance(action_space, gym.spaces.Discrete):
        return action
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        # Extract sizes of each dimension (e.g. [4, 2, 3])
        subs = action_space.nvec
        # Accumulate sizes to calculate offsets (e.g. [0, 4, 6, 9])
        # TODO Use initial=0 when moved to Python 3.8
        offsets = [0] + list(accumulate(subs, operator.add))[:-1]
        return action + np.array(offsets)
    elif isinstance(action_space, gym.spaces.Tuple):
        subs = n_vecs(action_space)
        offsets = [0] + list(accumulate(subs, operator.add))[:-1]
        return action + np.array(offsets)
    else:
        raise ActionSpaceNotSupported(action_space)


def _multicategorical_softmax(t: torch.Tensor, sizes: List[int]) -> torch.Tensor:
    probs = torch.empty_like(t)
    offset = 0
    for sub_space_size in sizes:
        probs[offset: offset + sub_space_size] = torch.softmax(
            t[offset: offset + sub_space_size], dim=-1
        )
        offset += sub_space_size
    return probs


def n_vecs(action_space: gym.spaces.Tuple) -> List[int]:
    nvecs = []
    for a_s in action_space.spaces:
        if isinstance(a_s, gym.spaces.Discrete):
            nvecs.append(a_s.n)
        elif isinstance(a_s, gym.spaces.MultiDiscrete):
            nvecs.extend(list(a_s.nvec))
        elif isinstance(a_s, gym.spaces.Tuple):
            nvecs.extend(n_vecs(a_s))
    return nvecs
