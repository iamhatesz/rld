import gym


class ActionSpaceNotSupported(Exception):
    def __init__(self, action_space: gym.Space):
        super().__init__(
            f"The action space `{action_space}` is not currently supported. The "
            f"currently supported action spaces are: Discrete, MultiDiscrete, Tuple."
        )
