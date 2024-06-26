import torch
import torch.nn as nn
import numpy as np

from .base_graph_nn import BaseGraphNN
from utility import create_batch
from .graph_rep.graph_structure import (
    graph_rep_add_data,
    graph_rep_add_connection,
    graph_rep_add_linear,
    graph_rep_add_image_data,
    graph_rep_add_flatten,
)


class ANN(BaseGraphNN):
    def __init__(
        self,
        in_features=64,
        w=8,
        h=8,
        out_features=10,
        dimensions=1,
        batch_size=5,
        loss_function=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        lr=0.05,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            batch_size=batch_size,
            lr=lr,
            name="ann",
            dataset_i=1,
        )
        self.w = w
        self.h = h

        # ----- Структура сети -------
        self.lin_1 = nn.Linear(in_features, 16)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(16, 16)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(16, out_features)
        self.softmax = nn.Softmax(dimensions)
        # ----------------------------

        self.set_optimizer(optimizer, lr)
        self.set_loss_function(loss_function)

    def forward(self, x):
        y = self.lin_1(x)
        y = self.relu_1(y)
        y = self.lin_2(y)
        y = self.relu_2(y)
        y = self.lin_3(y)
        y = self.softmax(y)
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
            self.curr_epoch += 1

        # Обновляем текущий loss_value
        self.loss_value = loss.detach().cpu().numpy().item()

    def graph_structure(self):
        structure = []

        # Добавляем изображение в качестве 1 слоя, чтобы пользователю было понятно, что это за датасет.
        structure.extend(
            graph_rep_add_image_data(
                [1, self.w, self.h], np.zeros((1, self.w, self.h)).tolist()
            )
        )
        structure.extend(graph_rep_add_connection(np.zeros((1, 1)).tolist(), False))
        structure.extend(graph_rep_add_flatten())
        structure.extend(
            graph_rep_add_connection(np.zeros((self.in_features, 1)).tolist(), False)
        )

        # Входной слой.
        structure.extend(graph_rep_add_data(self.in_features, [0] * self.in_features))

        # Структура сети.
        layer = self.lin_1
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(
            graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0])
        )

        layer = self.lin_2
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(
            graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0])
        )

        layer = self.lin_3
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "Softmax"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(self.out_features, [0] * self.out_features))

        return [{"model": self.name, "loss": self.loss_value, "structure": structure}]

    def forward_graph(self, data):
        # Добавляем дополнительное измерение.
        # Индексы слоев получаем из graph_structure.
        image_data = data.reshape(1, 1, self.w, self.h)
        data = data.unsqueeze(0)

        data_states = []
        data_states.append({"graphLayerIndex": 0, "w": image_data.squeeze(0).tolist()})
        data_states.append({"graphLayerIndex": 4, "w": data.squeeze(0).tolist()})

        y = self.lin_1(data)
        y = self.relu_1(y)
        data_states.append({"graphLayerIndex": 10, "w": y.squeeze(0).tolist()})

        y = self.lin_2(y)
        y = self.relu_2(y)
        data_states.append({"graphLayerIndex": 16, "w": y.squeeze(0).tolist()})

        y = self.lin_3(y)
        y = self.softmax(y)
        data_states.append({"graphLayerIndex": 22, "w": y.squeeze(0).tolist()})

        return data_states

    def backward_graph_batch(self, train_dataset):
        weights_states = [
            {"graphLayerIndex": 5, "w": self.lin_1.weight.tolist()},
            {"graphLayerIndex": 6, "w": self.lin_1.bias.tolist()},
            {"graphLayerIndex": 11, "w": self.lin_2.weight.tolist()},
            {"graphLayerIndex": 12, "w": self.lin_2.bias.tolist()},
            {"graphLayerIndex": 17, "w": self.lin_3.weight.tolist()},
            {"graphLayerIndex": 18, "w": self.lin_3.bias.tolist()},
        ]

        return {
            "dataIndex": 0,
            "layerIndex": 0,
            "weights": list(reversed(weights_states)),
        }
