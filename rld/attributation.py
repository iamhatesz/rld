import sys
from enum import IntEnum

from rld.model import Model
from rld.rollout import Rollout


class AttributationTarget(IntEnum):
    # The target equals to the actions picked by an agent
    PICKED = 0
    # The target chosen as argmax from action distribution
    HIGHEST = 1
    # 3 highest targets (calculated in the same way as `HIGHEST`)
    TOP3 = 3
    # 5 highest targets (calculated in the same way as `HIGHEST`)
    TOP5 = 5
    # All available targets
    ALL = sys.maxsize


def attribute_rollout(
    rollout: Rollout,
    model: Model,
    targets: AttributationTarget = AttributationTarget.PICKED,
) -> Rollout:
    raise NotImplementedError()
