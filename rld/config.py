from dataclasses import dataclass

from rld.attributation import AttributationTarget
from rld.model import Model
from rld.typing import BaselineBuilder


@dataclass
class Config:
    model: Model
    baseline: BaselineBuilder
    target: AttributationTarget
