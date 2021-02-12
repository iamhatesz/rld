from typing import Union, Sequence, Callable, Dict, List

import gym
import numpy as np
import torch

ObsLike = Union[np.ndarray, Dict[str, np.ndarray]]
ObsLikeStrict = np.ndarray
ObsTensorLike = Union[torch.Tensor, Dict[str, torch.Tensor]]
ObsTensorStrict = torch.Tensor
ObsBatchLike = Sequence[ObsLike]
ObsBatchLikeStrict = Sequence[np.ndarray]

ActionLike = Union[int, np.ndarray]
ActionBatchLike = Sequence[ActionLike]

RewardLike = float
RewardBatchLike = Sequence[RewardLike]

DoneLike = bool
DoneBatchLike = Sequence[DoneLike]

InfoValueLike = Union[str, int, float, bool, np.ndarray]
InfoLike = Dict[str, InfoValueLike]
InfoBatchLike = Sequence[InfoLike]

HiddenState = np.ndarray
HiddenStateTensor = torch.Tensor

AttributationLike = Union[np.ndarray, Dict[str, np.ndarray]]
AttributationLikeStrict = np.ndarray
AttributationBatchLike = Sequence[AttributationLike]

BaselineBuilder = Callable[[ObsLike], np.ndarray]

MultiDiscreteAsTuple = gym.spaces.Tuple
