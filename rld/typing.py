from typing import Union, Mapping, Sequence

import numpy as np
import torch

ObsLike = Union[np.ndarray, Mapping[str, np.ndarray]]
ObsBatchLike = Sequence[ObsLike]
ObsBatchLikeStrict = Sequence[np.ndarray]


def is_obs_dict(obs: Union[ObsLike, ObsBatchLike]) -> bool:
    if isinstance(obs, dict):
        return True
    return False


ActionLike = Union[int, np.ndarray]
ActionBatchLike = Sequence[ActionLike]


def is_action_array(action: ActionLike) -> bool:
    if isinstance(action, np.ndarray):
        return True
    return False


RewardLike = float
RewardBatchLike = Sequence[RewardLike]

DoneLike = bool
DoneBatchLike = Sequence[DoneLike]

InfoValueLike = Union[str, int, float, bool, np.ndarray]
InfoLike = Mapping[str, InfoValueLike]
InfoBatchLike = Sequence[InfoLike]
