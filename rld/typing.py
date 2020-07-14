from typing import Union, Mapping, Sequence

import numpy as np


ObsLike = Union[np.ndarray, Mapping[str, np.ndarray]]
ObsBatchLike = Union[np.ndarray, Mapping[str, np.ndarray]]


def is_obs_dict(obs: Union[ObsLike, ObsBatchLike]) -> bool:
    if isinstance(obs, dict):
        return True
    return False


ActionLike = Union[int, np.ndarray]
ActionBatchLike = np.ndarray


def is_action_array(action: ActionLike) -> bool:
    if isinstance(action, np.ndarray):
        return True
    return False


RewardLike = float
RewardBatchLike = np.ndarray

DoneLike = bool
DoneBatchLike = np.ndarray

InfoValueLike = Union[str, int, float, bool, np.ndarray]
InfoLike = Mapping[str, InfoValueLike]
InfoBatchLike = Sequence[InfoLike]
