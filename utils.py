import torch
import torch.nn as nn


# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    # bias의 형태로 학습할 수 있도록 만들어 놓음
    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)       # t()는 transpose. 첫 dim을 row의 형태로 만들어 주기 위함
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
