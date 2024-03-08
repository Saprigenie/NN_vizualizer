import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from math import floor


from config import DEVICE
from .utility import get_layers, graph_rep_add_data, graph_rep_add_connection, graph_rep_add_linear
from .utility import graph_rep_add_image_data, graph_rep_add_conv2d, graph_rep_add_maxpool2d, graph_rep_add_flatten

class CNN(nn.Module):
    def __init__(self, channels = 1, w = 8, h = 8, out_features = 10,  num_filters = 4, dimensions = 1):
        super().__init__()
        self.channels = channels
        self.w = w
        self.h = h
        self.out_features = out_features
        self.dim = dimensions

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
    

    def graph_structure(self):
        structure = []
        # Получаем слои нейронной сети.
        layers = get_layers(self)

        # Входной слой.
        structure.extend(graph_rep_add_image_data([1, self.w, self.h], np.zeros((1, self.w, self.h)).tolist()))

        # Структура сверточной части сети.
        layer = layers[0]
        structure.extend(graph_rep_add_connection(np.zeros((self.num_filters, 1)).tolist(), False))
        structure.extend(graph_rep_add_conv2d(layer))
        structure.extend(graph_rep_add_connection([0] * self.num_filters, False))

        # Промежуточные данные.
        shape_data = [self.num_filters, self.w_after * 2, self.h_after * 2]
        structure.extend(graph_rep_add_image_data(shape_data, np.zeros(shape_data).tolist()))

        layer = layers[1]
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
        layer = layers[2]
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = layers[3]
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "Softmax"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(self.out_features, [0] * self.out_features))

        return structure

    def train_epoch(self, loss_function, optimizer, train_loader):
        # Создаем вспомогательные списки для данных:
        losses_train = []

        # Обучение (цикл по батчам):
        for iteration, (X_batch, y_batch) in enumerate(train_loader):
            # Переводим модель в режим обучения:
            self.train()
            # Обнуляем градиенты у оптимизатора:
            optimizer.zero_grad()
            # Пропускам данные через модель:
            outputs = self(X_batch.to(DEVICE))\
            # Считаем функцию потерь:
            loss = loss_function(outputs, y_batch.to(DEVICE))
            # Делаем шаг в обратном направлении:
            loss.backward()
            # Собираем функцию потерь:
            losses_train.append(loss.detach().cpu().numpy().item())
            # Делаем шаг оптимизатора:
            optimizer.step()

        return losses_train

    def val_epoch(self, loss_function, val_loader, accuracy):
        # Создаем вспомогательные списки для данных:
        acc_val = []

        # Валидация (цикл по батчам):
        for iteration, (X_batch, y_batch) in enumerate(val_loader):
            # Подключаем режим валидации или тестирования:
            self.eval()
            # Отключаем рассчет градиентов:
            with torch.no_grad():
                outputs = self(X_batch.to(DEVICE))
                loss = loss_function(outputs, y_batch.to(DEVICE).to(torch.int64))
                # Получаем выбор модели:
                pred_classes = torch.argmax(outputs, 1)
                # Считаем метрику:
                batch_acc = accuracy(pred_classes, y_batch.to(DEVICE))
                curr_accuracy = batch_acc.detach().cpu().numpy().item()
                acc_val.append(curr_accuracy)

        return acc_val
