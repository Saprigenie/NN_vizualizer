import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from math import floor

from .utility import create_batch, graph_rep_add_data, graph_rep_add_connection, graph_rep_add_linear
from .utility import graph_rep_add_image_data, graph_rep_add_conv2d, graph_rep_add_maxpool2d, graph_rep_add_flatten


class CNN(nn.Module):
    def __init__(self, channels = 1, w = 8, h = 8, out_features = 10,  num_filters = 4, dimensions = 1, batch_size = 1):
        super().__init__()
        self.channels = channels
        self.w = w
        self.h = h
        self.out_features = out_features
        self.dim = dimensions
        self.batch_size = batch_size
        self.train_i = 0
        self.forward_i = 0
        self.state_forward = True 

        # kernel_size, stride и padding для Conv2d слоев.
        self.num_filters = num_filters
        self.k_size = 3
        self.s = 1
        self.p = 0

        # CNN сверточные слои.
        self.conv_1 = nn.Conv2d(in_channels = channels,
                                out_channels = channels*self.num_filters,
                                kernel_size = self.k_size,
                                stride = self.s,
                                padding = self.p)
        self.pool_1 = nn.MaxPool2d(2, 2)

        # Вычисление размера извлеченных признаков после Conv и MaxPool2d.
        self.w_after = self.compute_size_after_conv(self.w)
        self.h_after = self.compute_size_after_conv(self.h)

        # Полносвязанная нейронная сеть (классификатор).
        self.lin_1 = nn.Linear(self.channels*self.num_filters * self.w_after * self.h_after, 16)
        self.lin_2 = nn.Linear(16, out_features)

        self.relu = F.relu
        self.softmax = F.softmax

        # Задаем функцию потерь:
        self.loss_function = nn.CrossEntropyLoss()
        # Задаем оптимизатор:
        self.optimizer = torch.optim.SGD(self.parameters(), lr=.042)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def compute_size_after_conv(self, width_or_height):
        # Конкретно для архитектуры с 2-мя Conv2d и 2-мя MaxPool(2, 2).
        result = floor((width_or_height + 2*self.p - self.k_size) / self.s) + 1
        result = int(result / 2)

        return result


    def forward(self, x):
        # Так как изначально в датасете картинки преобразованы в одну линию, то нужно вернуть из обратно.
        x = x.reshape(-1, 1, self.w, self.h)

        y = self.relu(self.conv_1(x))
        y = self.pool_1(y)

        # Получаем размерность (размер батча, (out_channels последнего conv2d) * self.w_after * self.h_after)
        y = y.view(-1, self.channels*self.num_filters * self.w_after * self.h_after)
        y = self.relu(self.lin_1(y))
        y = self.lin_2(y)
        y = self.softmax(y, dim=self.dim)
        return y
    
    def train_batch(self, train_dataset):
        x_batch, y_batch = create_batch(train_dataset, self.train_i, self.batch_size)

        # Обучение:
        # Переводим модель в режим обучения:
        self.train()
        # Обнуляем градиенты у оптимизатора:
        self.optimizer.zero_grad()
        # Пропускам данные через модель:
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
        structure.extend(graph_rep_add_image_data([1, self.w, self.h], np.zeros((1, self.w, self.h)).tolist()))

        # Структура сверточной части сети.
        layer = self.conv_1
        structure.extend(graph_rep_add_connection(np.zeros((self.num_filters, 1)).tolist(), False))
        structure.extend(graph_rep_add_conv2d(layer))
        structure.extend(graph_rep_add_connection([0] * self.num_filters, False))

        # Промежуточные данные.
        shape_data = [self.num_filters, self.w_after * 2, self.h_after * 2]
        structure.extend(graph_rep_add_image_data(shape_data, np.zeros(shape_data).tolist()))

        layer =  self.pool_1
        structure.extend(graph_rep_add_connection([0] * self.num_filters, False))
        structure.extend(graph_rep_add_maxpool2d(layer, self.num_filters))
        structure.extend(graph_rep_add_connection([0] * self.num_filters, False))

        # Промежуточные данные.
        shape_data = [self.num_filters, self.w_after, self.h_after]
        structure.extend(graph_rep_add_image_data(shape_data, np.zeros(shape_data).tolist()))

        # Объединяем изображения и Flatten.
        structure.extend(graph_rep_add_connection(np.zeros((1, self.num_filters)).tolist(), False))
        structure.extend(graph_rep_add_flatten())

        # Промежуточные данные.
        count_data = self.channels*self.num_filters * self.w_after * self.h_after
        structure.extend(graph_rep_add_connection(np.zeros((count_data, 1)).tolist(), False))
        structure.extend(graph_rep_add_data(count_data, [0] * count_data))

        # Структура полносвязной части сети.
        layer = self.lin_1
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = self.lin_2
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "Softmax"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(self.out_features, [0] * self.out_features))

        return [{
            "model": "cnn",
            "structure": structure
        }]
    
    def forward_graph(self, data):
        # Превращаем в изображение.
        data = data.reshape(1, 1, self.w, self.h)

        data_states = []
        data_states.append({
            "graphLayerIndex": 0,
            "w": data.squeeze(0).tolist()
        })

        y = self.relu(self.conv_1(data))
        data_states.append({
            "graphLayerIndex": 6,
            "w": y.squeeze(0).tolist()
        })

        y = self.pool_1(y)
        data_states.append({
            "graphLayerIndex": 10,
            "w": y.squeeze(0).tolist()
        })

        # Получаем размерность (размер батча, (out_channels последнего conv2d) * self.w_after * self.h_after)
        y = y.view(-1, self.channels*self.num_filters * self.w_after * self.h_after)
        data_states.append({
            "graphLayerIndex": 14,
            "w": y.squeeze(0).tolist()
        })

        y = self.relu(self.lin_1(y))
        data_states.append({
            "graphLayerIndex": 20,
            "w": y.squeeze(0).tolist()
        })

        y = self.lin_2(y)
        y = self.softmax(y, dim=self.dim)
        data_states.append({
            "graphLayerIndex": 26,
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
            "model": "cnn",
            "type": "forward",
            "dataIndex": 0,
            "layerIndex": 0,
            "ended": False,
            "weights": [self.forward_graph(data) for data in x_batch]
        }

    def backward_graph_batch(self):
        weights_states = [{
            "graphLayerIndex": 2,
            "w": [self.conv_1.weight.tolist(), self.conv_1.bias.tolist()]
        }, {
            "graphLayerIndex": 15,
            "w": self.lin_1.weight.tolist()
        }, {
            "graphLayerIndex": 16,
            "w": self.lin_1.bias.tolist()
        }, {
            "graphLayerIndex": 21,
            "w": self.lin_2.weight.tolist()
        }, {
            "graphLayerIndex": 22,
            "w": self.lin_2.bias.tolist()
        }]

        self.state_forward = True

        return {
            "model": "cnn",
            "type": "backward",
            "layerIndex": 0,
            "ended": False,
            "weights": list(reversed(weights_states))
        }
