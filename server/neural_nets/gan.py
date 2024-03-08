import torch
import torch.nn as nn
from torch.nn import functional as F

from config import DEVICE
from .utility import get_layers, graph_rep_add_data, graph_rep_add_connection, graph_rep_add_linear


class GANgenerator(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.lin_1 = nn.Linear(in_features, 16)
        self.lin_2 = nn.Linear(16, 32)
        self.lin_3 = nn.Linear(32, out_features)
        self.relu = F.relu

    def forward(self, x):
        y = self.relu(self.lin_1(x))
        y = self.relu(self.lin_2(y))
        y = self.lin_3(y)
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
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(self.out_features, [0] * self.out_features))

        return structure


class GANdiscriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.lin_1 = nn.Linear(in_features, 32)
        self.lin_2 = nn.Linear(32, 16)
        self.lin_3 = nn.Linear(16, 1)
        self.relu = F.relu
        self.sigmoid = F.sigmoid

    def forward(self, x):
        y = self.relu(self.lin_1(x))
        y = self.relu(self.lin_2(y))
        y = self.sigmoid(self.lin_3(y))
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
        structure.extend(graph_rep_add_linear(layer, "Sigmoid"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        return structure
    

class GAN:
    def __init__(self, in_vector_size = 100, img_size = 64):
        super().__init__()
        self.in_vector_size = in_vector_size

        # Инициализируем генератор и дискриминатор GAN-а.
        self.generator = GANgenerator(in_vector_size, img_size)
        self.discriminator = GANdiscriminator(img_size)

    def graph_structure(self):
        return {
            "generator": self.generator.graph_structure(),
            "discriminator": self.discriminator.graph_structure()
        }

    def train_epoch(self, loss_f, optimizer_g, optimizer_d, train_loader):
        # Создаем вспомогательные списки для данных:
        losses_generator = []
        losses_discriminator = []

        # Обучение (цикл по батчам):
        for iteration, (x_batch, y_batch) in enumerate(train_loader):
            ## Тренируем генератор
            # Создаем шум для генератора:
            x_noise = torch.randn(x_batch.shape[0], self.in_vector_size, device=DEVICE)
            # Переводим генератор в режим обучения:
            self.generator.train()
            # Обнуляем градиенты у оптимизатора:
            optimizer_g.zero_grad()
            # Генерируем изображения:
            generated_imgs = self.generator(x_noise)
            # Классифицируем сгенерированные изображения дискриминатором.
            d_output = self.discriminator(generated_imgs)
            # Мы хотим максимизировать потери дискриминатора, поэтому
            # считаем, что у сгенерированных изображений лейбел 1
            # (что они не сгенерированны).
            loss_g = loss_f(d_output, torch.ones((x_batch.shape[0], 1), device=DEVICE))
            # Делаем шаг в обратном направлении:
            loss_g.backward()
            # Собираем функцию потерь:
            losses_generator.append(loss_g.detach().cpu().numpy().item())
            # Делаем шаг оптимизатора:
            optimizer_g.step()

            ## Тренируем дискриминатор:
            # Переводим дискриминатор в режим обучения:
            self.discriminator.train()
            # Обнуляем градиенты у дискриминатора:
            optimizer_d.zero_grad()
            # Генерируем лживые изображения для дискриминатора:
            generated_imgs = self.generator(x_noise)
            # Пропускам данные через модель:
            input_d = torch.cat([x_batch.to(DEVICE), generated_imgs], dim=0)
            outputs = self.discriminator(input_d)
            # Создаем метки (1 - настоящее изображение, 0 - сгенерированное)
            y_generator = torch.cat([torch.ones((x_batch.shape[0], 1), device=DEVICE),
                                        torch.zeros((x_batch.shape[0], 1), device=DEVICE)], dim=0)
            # Считаем функцию потерь:
            loss_d = loss_f(outputs, y_generator)
            # Делаем шаг в обратном направлении:
            loss_d.backward()
            # Собираем функцию потерь:
            losses_discriminator.append(loss_d.detach().cpu().numpy().item())
            # Делаем шаг оптимизатора:
            optimizer_d.step()

        return (losses_generator, losses_discriminator)