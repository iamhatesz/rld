from typing import Union, Mapping, Sequence, Callable

import numpy as np
import torch

ObsLike = Union[np.ndarray, Mapping[str, np.ndarray]]
ObsLikeStrict = np.ndarray
ObsBatchLike = Sequence[ObsLike]
ObsBatchLikeStrict = Sequence[np.ndarray]

ActionLike = Union[int, np.ndarray]
ActionBatchLike = Sequence[ActionLike]

RewardLike = float
RewardBatchLike = Sequence[RewardLike]

DoneLike = bool
DoneBatchLike = Sequence[DoneLike]

InfoValueLike = Union[str, int, float, bool, np.ndarray]
InfoLike = Mapping[str, InfoValueLike]
InfoBatchLike = Sequence[InfoLike]

AttributationLike = Union[np.ndarray, Mapping[str, np.ndarray]]
AttributationLikeStrict = torch.Tensor
AttributationBatchLike = Sequence[AttributationLike]

BaselineBuilder = Callable[[ObsLike], np.ndarray]
