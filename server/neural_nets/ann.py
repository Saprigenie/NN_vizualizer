import torch
import torch.nn as nn
from torch.nn import functional as F

from config import DEVICE
from .utility import get_layers, graph_rep_add_data, graph_rep_add_connection, graph_rep_add_linear


class ANN(nn.Module):
    def __init__(self, in_features = 64, out_features = 10, dimensions = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dimensions

        self.lin_1 = nn.Linear(in_features, 16)
        self.lin_2 = nn.Linear(16, 16)
        self.lin_3 = nn.Linear(16, out_features)
        self.relu = F.relu
        self.softmax = F.softmax

    def forward(self, x):
        y = self.relu(self.lin_1(x))
        y = self.relu(self.lin_2(y))
        y = self.lin_3(y)
        y = self.softmax(y, dim=self.dim)
        return y
    
    def graph_structure(self):
        structure = []
        # Получаем слои нейронной сети.
        layers = get_layers(self)

        # Входной слой.
        structure.extend(graph_rep_add_data(self.in_features, [0] * self.in_features))

        # Структура сети.
        layer = layers[0]
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = layers[1]
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = layers[2]
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
