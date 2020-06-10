import torch
from typing import Union, NewType

FloatTensor = NewType("FloatTensor", Union[torch.FloatTensor, torch.nn.Parameter])
