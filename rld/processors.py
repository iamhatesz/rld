from gym.spaces import unflatten, flatten
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.utils.torch_ops import convert_to_non_torch_type

from rld.model import RayModel
from rld.typing import (
    ObsLike,
    ObsLikeStrict,
    AttributationLikeStrict,
    AttributationLike,
)


class ObsPreprocessor:
    def transform(self, obs: ObsLike) -> ObsLikeStrict:
        raise NotImplementedError


class NoObsPreprocessor:
    def transform(self, obs: ObsLike) -> ObsLikeStrict:
        return obs


class RayObsPreprocessor(ObsPreprocessor):
    def __init__(self, preprocessor: Preprocessor):
        self.preprocessor = preprocessor

    def transform(self, obs: ObsLike) -> ObsLikeStrict:
        return self.preprocessor.transform(obs)


class AttributionPostprocessor:
    def transform(self, attribution: AttributationLikeStrict) -> AttributationLike:
        raise NotImplementedError


class RayAttributionPostprocessor(AttributionPostprocessor):
    def __init__(self, model: RayModel):
        self.model = model

    def transform(self, attribution: AttributationLikeStrict) -> AttributationLike:
        attribution = restore_original_dimensions(
            attribution, self.model.obs_space, tensorlib="torch"
        )
        attribution = convert_to_non_torch_type(attribution)
        if hasattr(self.model.obs_space, "original_space"):
            obs_space = self.model.obs_space.original_space
        else:
            obs_space = self.model.obs_space
        attribution = unflatten(obs_space, flatten(obs_space, attribution))
        return attribution
