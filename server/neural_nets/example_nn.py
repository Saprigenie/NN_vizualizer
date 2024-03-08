import torch
import torch.nn as nn
from torch.nn import functional as F


class ExapmleNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Слои нейронной сети, которая принимает in_features параметров и
        # Выдает out_features параметров.

    def forward(self, x):
        # Реализация прямого прохода по слоям нейронной сети.
        pass

    def graph_structure():
        # Возвращает список словарей, в котором разбита структура сети по слоям.
        # Например:
        """
        [
            {
                "type": "Data",
                "count": 64,
                "weights": [...]
            },
            {
                "type": "Connections",
                "displayWeights": True
                "weights": [[...], [...], ...]
            },
            {
                "type": "Linear",
                "count": 20,
                "weights": [...]
            },
            {
                "type": "Activation",
                "count": 20,
                "activation": "ReLU" 
            },
            ...
        ]
        """
        pass

    def train_epoch(self, loss_function, optimizer):
        # 1 эпоха обучения нейронной сети.
        pass

    def val_epoch(self, loss_function):
        # 1 эпоха валидации нейронной сети.
        pass
