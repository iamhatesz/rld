from dataclasses import dataclass
from typing import Optional, Union

from rld.attributation import AttributationTarget, AttributationNormalizationMode
from rld.model import Model
from rld.typing import BaselineBuilder


@dataclass
class Config:
    model: Model
    baseline: BaselineBuilder
    target: AttributationTarget = AttributationTarget.PICKED
    normalize_sign: AttributationNormalizationMode = AttributationNormalizationMode.ALL
    normalize_obs_image_channel_dim: Optional[int] = None
    normalize_outlier_percentile: Union[int, float] = 5
