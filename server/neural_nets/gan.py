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
    graph_rep_add_reshape,
)


class GANgenerator(BaseGraphNN):
    def __init__(
        self,
        in_features,
        out_features,
        w=8,
        h=8,
        batch_size=5,
        optimizer=torch.optim.SGD,
        lr=0.05,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            batch_size=batch_size,
            lr=lr,
            name="generator",
            dataset_i=1,
        )
        self.w = w
        self.h = h

        # ----- Структура сети -------
        self.lin_1 = nn.Linear(in_features, 16)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(16, 32)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(32, out_features)
        # ----------------------------

        self.set_optimizer(optimizer, lr)

    def forward(self, x):
        y = self.lin_1(x)
        y = self.relu_1(y)
        y = self.lin_2(y)
        y = self.relu_2(y)
        y = self.lin_3(y)
        return y

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
        structure.extend(graph_rep_add_linear(layer, None))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(self.out_features, [0] * self.out_features))

        # Добавляем изображение в качестве 1 слоя, чтобы пользователю было понятно, что это за датасет.
        structure.extend(
            graph_rep_add_connection(np.zeros((1, self.out_features)).tolist(), False)
        )
        structure.extend(graph_rep_add_reshape())
        structure.extend(graph_rep_add_connection(np.zeros((1, 1)).tolist(), False))
        structure.extend(
            graph_rep_add_image_data(
                [1, self.w, self.h], np.zeros((1, self.w, self.h)).tolist()
            )
        )

        return {"model": self.name, "loss": self.loss_value, "structure": structure}

    def forward_graph(self, data):
        # Добавляем дополнительное измерение.
        data = data.unsqueeze(0)

        data_states = []
        data_states.append({"graphLayerIndex": 0, "w": data.squeeze(0).tolist()})

        y = self.lin_1(data)
        y = self.relu_1(y)
        data_states.append({"graphLayerIndex": 6, "w": y.squeeze(0).tolist()})

        y = self.lin_2(y)
        y = self.relu_2(y)
        data_states.append({"graphLayerIndex": 12, "w": y.squeeze(0).tolist()})

        y = self.lin_3(y)
        data_states.append({"graphLayerIndex": 16, "w": y.squeeze(0).tolist()})

        image_data = y.reshape(1, 1, self.w, self.h)
        data_states.append({"graphLayerIndex": 20, "w": image_data.squeeze(0).tolist()})

        return data_states

    def forward_graph_batch(self, train_dataset, train_i):
        x_batch, _ = create_batch(train_dataset, train_i, self.batch_size)
        # Подменяем данные на рандомный тензор, так как это стандартный вход для генератора.
        x_batch = torch.randn(len(x_batch), self.in_features)

        return {
            "dataIndex": 0,
            "layerIndex": 0,
            "weights": [self.forward_graph(data) for data in x_batch],
        }

    def backward_graph_batch(self, train_dataset):
        weights_states = [
            {"graphLayerIndex": 1, "w": self.lin_1.weight.tolist()},
            {"graphLayerIndex": 2, "w": self.lin_1.bias.tolist()},
            {"graphLayerIndex": 7, "w": self.lin_2.weight.tolist()},
            {"graphLayerIndex": 8, "w": self.lin_2.bias.tolist()},
            {"graphLayerIndex": 13, "w": self.lin_3.weight.tolist()},
            {"graphLayerIndex": 14, "w": self.lin_3.bias.tolist()},
        ]

        return {
            "dataIndex": 0,
            "layerIndex": 0,
            "weights": list(reversed(weights_states)),
        }

    def graph_batch(self, forward_weights, backward_weights, train_dataset, train_i):
        return self.form_train_state(
            "forward", forward_weights, backward_weights, len(train_dataset), train_i
        )


class GANdiscriminator(BaseGraphNN):
    def __init__(
        self,
        in_features,
        w=8,
        h=8,
        out_features=1,
        batch_size=5,
        optimizer=torch.optim.SGD,
        lr=0.05,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            batch_size=batch_size
            * 2,  # Так как учится на двойном объеме данных (реальные/фейковые).
            lr=lr,
            name="discriminator",
            dataset_i=1,
        )
        self.w = w
        self.h = h

        # ----- Структура сети -------
        self.lin_1 = nn.Linear(in_features, 32)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(32, 16)
        self.relu_2 = nn.ReLU()
        self.lin_3 = nn.Linear(16, out_features)
        self.sigmoid = nn.Sigmoid()
        # ----------------------------

        # Задаем оптимизатор:
        self.set_optimizer(optimizer, lr)

    def set_batch_size(self, batch_size):
        return super().set_batch_size(batch_size * 2)

    def forward(self, x):
        y = self.lin_1(x)
        y = self.relu_1(y)
        y = self.lin_2(y)
        y = self.relu_2(y)
        y = self.lin_3(y)
        y = self.sigmoid(y)
        return y

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
        structure.extend(graph_rep_add_linear(layer, "Sigmoid"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(
            graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0])
        )

        return {"model": self.name, "loss": self.loss_value, "structure": structure}

    def forward_graph(self, data):
        # Добавляем дополнительное измерение.
        data = data.unsqueeze(0)
        image_data = data.reshape(1, 1, self.w, self.h)

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
        y = self.sigmoid(y)
        data_states.append({"graphLayerIndex": 22, "w": y.squeeze(0).tolist()})

        return data_states

    def forward_graph_batch(self, train_dataset, generator, train_i):
        # Половина батча - реальные данные, половина - фейковые.
        x_batch, _ = create_batch(train_dataset, train_i, self.batch_size // 2)
        x_noise = torch.randn(self.batch_size // 2, generator.in_features)
        x_gen_batch = generator(x_noise)

        x_batch = torch.cat((x_batch, x_gen_batch), dim=0)

        return {
            "dataIndex": 0,
            "layerIndex": 0,
            "weights": [self.forward_graph(data) for data in x_batch],
        }

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

    def graph_batch(self, forward_weights, backward_weights, train_dataset, train_i):
        return self.form_train_state(
            "forward",
            forward_weights,
            backward_weights,
            2 * len(train_dataset),
            2 * train_i,
        )


class GAN(BaseGraphNN):
    def __init__(self, in_vector_size=100, img_size=64, batch_size=5, lr=0.05):
        super().__init__(
            in_features=in_vector_size,
            out_features=1,
            batch_size=batch_size,
            lr=lr,
            name="GAN",
            dataset_i=1,
        )
        self.in_vector_size = in_vector_size

        # Кто обучается сейчас.
        self.generator_training = True

        # Инициализируем генератор и дискриминатор GAN-а.
        self.generator = GANgenerator(in_vector_size, img_size)
        self.discriminator = GANdiscriminator(img_size)

        # Задаем функцию потерь:
        self.loss_function = nn.BCELoss()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.generator.set_batch_size(batch_size)
        self.discriminator.set_batch_size(batch_size)

    def set_lr(self, lr):
        self.lr = lr
        self.generator.set_lr(lr)
        self.discriminator.set_lr(lr)

    def train_generator_batch(self):
        ## Тренируем генератор
        # Создаем шум для генератора:
        x_noise = torch.randn(self.batch_size, self.in_vector_size)
        # Переводим генератор в режим обучения:
        self.generator.train()
        # Обнуляем градиенты у оптимизатора:
        self.generator.optimizer.zero_grad()
        # Генерируем изображения:
        generated_imgs = self.generator(x_noise)
        # Классифицируем сгенерированные изображения дискриминатором.
        d_output = self.discriminator(generated_imgs)
        # Мы хотим максимизировать потери дискриминатора, поэтому
        # считаем, что у сгенерированных изображений лейбел 1
        # (что они не сгенерированны).
        loss_g = self.loss_function(d_output, torch.ones((self.batch_size, 1)))
        # Делаем шаг в обратном направлении:
        loss_g.backward()
        # Делаем шаг оптимизатора:
        self.generator.optimizer.step()

        # Обновляем текущий loss_value
        self.generator.loss_value = loss_g.detach().cpu().numpy().item()

    def train_discriminator_batch(self, train_dataset):
        x_batch, _ = create_batch(train_dataset, self.train_i, self.batch_size)
        # Создаем шум для генератора:
        x_noise = torch.randn(self.batch_size, self.in_vector_size)

        ## Тренируем дискриминатор:
        # Переводим дискриминатор в режим обучения:
        self.discriminator.train()
        # Обнуляем градиенты у дискриминатора:
        self.discriminator.optimizer.zero_grad()
        # Генерируем лживые изображения для дискриминатора:
        generated_imgs = self.generator(x_noise)
        # Пропускам данные через модель:
        input_d = torch.cat([x_batch, generated_imgs], dim=0)
        outputs = self.discriminator(input_d)
        # Создаем метки (1 - настоящее изображение, 0 - сгенерированное)
        y_generator = torch.cat(
            [torch.ones((x_batch.shape[0], 1)), torch.zeros((x_batch.shape[0], 1))],
            dim=0,
        )
        # Считаем функцию потерь:
        loss_d = self.loss_function(outputs, y_generator)
        # Делаем шаг в обратном направлении:
        loss_d.backward()
        # Делаем шаг оптимизатора:
        self.discriminator.optimizer.step()

        # Обновляем индекс данных, которые будем брать в следующий раз.
        self.train_i += len(x_batch)
        if self.train_i >= len(train_dataset):
            self.train_i = 0
            self.curr_epoch += 1

        # Обновляем текущий loss_value
        self.discriminator.loss_value = loss_d.detach().cpu().numpy().item()

    def train_batch(self, train_dataset):
        # Тренируем по очереди генератор и дискриминатор.
        if self.generator_training:
            self.train_generator_batch()
        else:
            self.train_discriminator_batch(train_dataset)

        self.generator_training = not self.generator_training

    def graph_structure(self):
        return [self.generator.graph_structure(), self.discriminator.graph_structure()]

    def graph_batch(self, train_dataset):
        # Узнаем, у кого сейчас обучение.
        if self.generator_training:
            forward_weights = self.generator.forward_graph_batch(
                train_dataset, self.train_i
            )
            self.train_batch(train_dataset)
            backward_weights = self.generator.backward_graph_batch(train_dataset)

            return self.generator.graph_batch(
                forward_weights, backward_weights, train_dataset, self.train_i
            )
        else:
            forward_weights = self.discriminator.forward_graph_batch(
                train_dataset, self.generator, self.train_i
            )
            self.train_batch(train_dataset)
            backward_weights = self.discriminator.backward_graph_batch(train_dataset)

            return self.discriminator.graph_batch(
                forward_weights, backward_weights, train_dataset, self.train_i
            )
