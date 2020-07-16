import torch

from rld.model import Model, RayModel


class RayModelWrapper(Model):
    def __init__(self, model: RayModel):
        super().__init__()
        self.model = model

    def unwrapped(self):
        return self.model

    def forward(self, x):
        input_dict = {"obs": x, "obs_flat": x}
        state = None
        seq_lens = None
        return self.model(input_dict, state, seq_lens)[0]

    def input_device(self) -> torch.device:
        return next(self.model.parameters()).device
