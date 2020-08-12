import gym

BOX_OBS_SPACE = gym.spaces.Box(-1, 1, shape=(6,))
IMAGE_OBS_SPACE = gym.spaces.Box(0, 1, shape=(84, 84, 4))
DICT_OBS_SPACE = gym.spaces.Dict(
    a=gym.spaces.Box(-1, 1, (4, 6)), b=gym.spaces.Box(-1, 1, (2,))
)
DISCRETE_ACTION_SPACE = gym.spaces.Discrete(4)
MULTI_DISCRETE_ACTION_SPACE = gym.spaces.MultiDiscrete([4, 3, 2])
TUPLE_ACTION_SPACE = gym.spaces.Tuple(
    (gym.spaces.MultiDiscrete([4, 2, 3]), gym.spaces.Discrete(2))
)
