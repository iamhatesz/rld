from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

BOX_OBS_SPACE = Box(-1, 1, shape=(6,))
IMAGE_OBS_SPACE = Box(0, 1, shape=(84, 84, 4))
DICT_OBS_SPACE = Dict(a=Box(-1, 1, (4, 6)), b=Box(-1, 1, (2,)))
DISCRETE_ACTION_SPACE = Discrete(4)
MULTI_DISCRETE_ACTION_SPACE = MultiDiscrete([4, 3, 2])
TUPLE_ACTION_SPACE = Tuple((Discrete(4), Discrete(3), Discrete(2)))
