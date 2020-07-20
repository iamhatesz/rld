from ray.rllib.models.preprocessors import Preprocessor

from rld.typing import (
    ObsLike,
    ObsLikeStrict,
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
