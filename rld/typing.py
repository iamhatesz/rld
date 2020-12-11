from typing import Union, Sequence, Callable, Dict, List

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

HiddenState = torch.Tensor

AttributationLike = Union[np.ndarray, Dict[str, np.ndarray]]
AttributationLikeStrict = torch.Tensor
AttributationBatchLike = Sequence[AttributationLike]

BaselineBuilder = Callable[[ObsLike], np.ndarray]
