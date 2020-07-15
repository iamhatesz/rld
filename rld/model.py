from abc import ABC

import torch
import torch.nn as nn
from ray.rllib.models.modelv2 import ModelV2


class Model(nn.Module):
    def input_device(self) -> torch.device:
        raise NotImplementedError

    def output_device(self) -> torch.device:
        return self.input_device()


class RayModel(ABC, Model, ModelV2):
    pass
