import gym


class ActionSpaceNotSupported(Exception):
    def __init__(self, action_space: gym.Space):
        super().__init__(
            f"The action space `{action_space}` is not currently supported. Currently supported action spaces: Discrete, MultiDiscrete, Tuple."
        )
