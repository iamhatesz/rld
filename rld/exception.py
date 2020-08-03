from enum import Enum
from pathlib import Path
from typing import Any, Type

import gym


class ActionSpaceNotSupported(Exception):
    def __init__(self, action_space: gym.Space):
        super().__init__(
            f"The action space `{action_space}` is not currently supported. The "
            f"currently supported action spaces are: Discrete, MultiDiscrete, Tuple."
        )


class EnumValueNotFound(Exception):
    def __init__(self, value: Any, enum: Type[Enum]):
        super().__init__(f"The value `{value}` was not found in enum `{enum}`")


class InvalidConfigProvided(Exception):
    def __init__(self, config_path: Path):
        super().__init__(
            f"The provided config file (`{config_path}`) does not contain "
            f"a `config` variable."
        )


class APIException(Exception):
    def __init__(self, message: str, status_code: int = 501):
        self.message = message
        self.status_code = status_code

    def as_dict(self) -> dict:
        return {"message": self.message}


class TrajectoryNotFound(APIException):
    def __init__(self):
        super().__init__("Trajectory not found.", status_code=404)


class EndpointNotFound(APIException):
    def __init__(self):
        super().__init__("Endpoint not found.", status_code=404)
