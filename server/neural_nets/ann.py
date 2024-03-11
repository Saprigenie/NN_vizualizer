import torch
import torch.nn as nn
from torch.nn import functional as F

from .utility import create_batch, graph_rep_add_data, graph_rep_add_connection, graph_rep_add_linear


class ANN(nn.Module):
    def __init__(self, in_features = 64, out_features = 10, dimensions = 1, batch_size = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dimensions
        self.batch_size = batch_size
        self.train_i = 0
        self.forward_i = 0
        self.state_forward = True 

        self.lin_1 = nn.Linear(in_features, 16)
        self.lin_2 = nn.Linear(16, 16)
        self.lin_3 = nn.Linear(16, out_features)
        self.relu = F.relu
        self.softmax = F.softmax

        # Задаем функцию потерь:
        self.loss_function = nn.CrossEntropyLoss()
        # Задаем оптимизатор:
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.042)

    def forward(self, x):
        y = self.relu(self.lin_1(x))
        y = self.relu(self.lin_2(y))
        y = self.lin_3(y)
        y = self.softmax(y, dim=self.dim)
        return y
    
    def train_batch(self, train_dataset):
        x_batch, y_batch = create_batch(train_dataset, self.train_i, self.batch_size)

        # Обучение:
        # Переводим модель в режим обучения:
        self.train()
        # Обнуляем градиенты у оптимизатора:
        self.optimizer.zero_grad()
        # Пропускаем данные через модель:
        outputs = self(x_batch)
        # Считаем функцию потерь:
        loss = self.loss_function(outputs, y_batch)
        # Делаем шаг в обратном направлении:
        loss.backward()
        # Делаем шаг оптимизатора:
        self.optimizer.step()

        # Обновляем индекс данных, которые будем брать в следующий раз.
        self.train_i += len(x_batch)
        if self.train_i >= len(train_dataset):
            self.train_i = 0
    
    def graph_structure(self):
        structure = []

        # Входной слой.
        structure.extend(graph_rep_add_data(self.in_features, [0] * self.in_features))

        # Структура сети.
        layer = self.lin_1
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = self.lin_2
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = self.lin_3
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "Softmax"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(self.out_features, [0] * self.out_features))

        return [{
            "model": "ann",
            "structure": structure
        }]
    
    def forward_graph(self, data):
        # Добавляем дополнительное измерение.
        # Индексы слоев получаем из graph_structure.
        data = data.unsqueeze(0)

        data_states = []
        data_states.append({
            "graphLayerIndex": 0,
            "w": data.squeeze(0).tolist()
        })

        y = self.relu(self.lin_1(data))
        data_states.append({
            "graphLayerIndex": 6,
            "w": y.squeeze(0).tolist()
        })

        y = self.relu(self.lin_2(y))
        data_states.append({
            "graphLayerIndex": 12,
            "w": y.squeeze(0).tolist()
        })

        y = self.lin_3(y)
        y = self.softmax(y, dim=self.dim)
        data_states.append({
            "graphLayerIndex": 18,
            "w": y.squeeze(0).tolist()
        })

        return data_states
    
    def forward_graph_batch(self, train_dataset):
        x_batch, _ = create_batch(train_dataset, self.forward_i, self.batch_size)

        # Обновляем индекс данных, которые будем брать в следующий раз.
        self.forward_i += len(x_batch)
        if self.forward_i >= len(train_dataset):
            self.forward_i = 0

        self.state_forward = False

        return {
            "model": "ann",
            "type": "forward",
            "dataIndex": 0,
            "layerIndex": 0,
            "ended": False,
            "weights": [self.forward_graph(data) for data in x_batch]
        }

    def backward_graph_batch(self):
        weights_states = [{
            "graphLayerIndex": 1,
            "w": self.lin_1.weight.tolist()
        }, {
            "graphLayerIndex": 2,
            "w": self.lin_1.bias.tolist()
        }, {
            "graphLayerIndex": 7,
            "w": self.lin_2.weight.tolist()
        }, {
            "graphLayerIndex": 8,
            "w": self.lin_2.bias.tolist()
        }, {
            "graphLayerIndex": 13,
            "w": self.lin_3.weight.tolist()
        }, {
            "graphLayerIndex": 14,
            "w": self.lin_3.bias.tolist()
        }]

        self.state_forward = True

        return {
            "model": "ann",
            "type": "backward",
            "layerIndex": 0,
            "ended": False,
            "weights": list(reversed(weights_states))
        }
