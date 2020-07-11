import torch.nn as nn


class RayToCaptumModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        input_dict = {"obs": x}
        state = None
        seq_lens = None
        return self.model(input_dict, state, seq_lens)[0]
