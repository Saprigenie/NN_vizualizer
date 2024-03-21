import torch
import torch.nn as nn

from .base_graph_nn import BaseGraphNN
from .utility.utility import create_batch
from .utility.graph_structure import graph_rep_add_data, graph_rep_add_connection, graph_rep_add_linear


class GANgenerator(BaseGraphNN):
    def __init__(self, in_features, out_features, batch_size = 1, optimizer = torch.optim.SGD, lr = 0.042):
        super().__init__(
            in_features=in_features, 
            out_features = out_features, 
            batch_size = batch_size
        )

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
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = self.lin_2
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, "ReLU"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Промежуточные данные.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        layer = self.lin_3
        structure.extend(graph_rep_add_connection(layer.weight.tolist()))
        structure.extend(graph_rep_add_linear(layer, None))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(self.out_features, [0] * self.out_features))

        return {
            "model": "generator",
            "structure": structure
        }
    
    def forward_graph(self, data):
        # Добавляем дополнительное измерение.
        data = data.unsqueeze(0)

        data_states = []
        data_states.append({
            "graphLayerIndex": 0,
            "w": data.squeeze(0).tolist()
        })

        y = self.lin_1(data)
        y = self.relu_1(y)
        data_states.append({
            "graphLayerIndex": 6,
            "w": y.squeeze(0).tolist()
        })

        y = self.lin_2(y)
        y = self.relu_2(y)
        data_states.append({
            "graphLayerIndex": 12,
            "w": y.squeeze(0).tolist()
        })

        y = self.lin_3(y)
        data_states.append({
            "graphLayerIndex": 16,
            "w": y.squeeze(0).tolist()
        })

        return data_states
    
    def forward_graph_batch(self, train_dataset):
        x_batch, _ = create_batch(train_dataset, self.forward_i, self.batch_size)
        # Подменяем данные на рандомный тензор, так как это стандартный вход для генератора.
        x_batch = torch.randn(len(x_batch), self.in_features)

        # Обновляем индекс данных, которые будем брать в следующий раз.
        self.forward_i += len(x_batch)
        if self.forward_i == len(train_dataset):
            self.forward_i = 0

        return {
            "model": "generator",
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

        return {
            "model": "generator",
            "type": "backward",
            "layerIndex": 0,
            "ended": False,
            "weights": list(reversed(weights_states))
        }


class GANdiscriminator(BaseGraphNN):
    def __init__(self, in_features, out_features = 1, batch_size = 1, optimizer = torch.optim.SGD, lr = 0.042):
        super().__init__(
            in_features=in_features, 
            out_features = out_features, 
            batch_size = batch_size
        )

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
        structure.extend(graph_rep_add_linear(layer, "Sigmoid"))
        structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

        # Выходной слой.
        structure.extend(graph_rep_add_data(layer.weight.shape[0], [0] * layer.weight.shape[0]))

        return {
            "model": "discriminator",
            "structure": structure
        }
    
    def forward_graph(self, data):
        # Добавляем дополнительное измерение.
        data = data.unsqueeze(0)

        data_states = []
        data_states.append({
            "graphLayerIndex": 0,
            "w": data.squeeze(0).tolist()
        })

        y = self.lin_1(data)
        y = self.relu_1(y)
        data_states.append({
            "graphLayerIndex": 6,
            "w": y.squeeze(0).tolist()
        })

        y = self.lin_2(y)
        y = self.relu_2(y)
        data_states.append({
            "graphLayerIndex": 12,
            "w": y.squeeze(0).tolist()
        })

        y = self.lin_3(y)
        y = self.sigmoid(y)
        data_states.append({
            "graphLayerIndex": 18,
            "w": y.squeeze(0).tolist()
        })

        return data_states
    
    def forward_graph_batch(self, train_dataset, generator):
        x_batch, _ = create_batch(train_dataset, self.forward_i, self.batch_size)

        x_noise = torch.randn(self.batch_size, generator.in_features)
        x_gen_batch = generator(x_noise)

        x_batch = torch.cat((x_batch, x_gen_batch), dim=0)

        # Обновляем индекс данных, которые будем брать в следующий раз.
        self.forward_i += len(x_batch)
        if self.forward_i >= len(train_dataset) * 2:
            self.forward_i = 0

        return {
            "model": "discriminator",
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

        return {
            "model": "discriminator",
            "type": "backward",
            "layerIndex": 0,
            "ended": False,
            "weights": list(reversed(weights_states))
        }


class GAN:
    def __init__(self, in_vector_size = 100, img_size = 64, batch_size = 1):
        super().__init__()
        self.in_vector_size = in_vector_size
        self.batch_size = batch_size
        self.train_i = 0
        self.state_forward = True 
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
        y_generator = torch.cat([torch.ones((x_batch.shape[0], 1)),
                                torch.zeros((x_batch.shape[0], 1))], dim=0)
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

    def train_batch(self, train_dataset):
        # Тренируем по очереди генератор и дискриминатор.
        if self.generator_training:
            self.train_generator_batch()
        else:
            self.train_discriminator_batch(train_dataset)

        self.generator_training = not self.generator_training

    def graph_structure(self):
        return [
            self.generator.graph_structure(),
            self.discriminator.graph_structure()
        ]
    
    def forward_graph_batch(self, train_dataset):
        self.state_forward = False

        # Узнаем, у кого сейчас форвард по флагу.
        if self.generator_training:
            return self.generator.forward_graph_batch(train_dataset)
        else:
            return self.discriminator.forward_graph_batch(train_dataset, self.generator)

    def backward_graph_batch(self):
        self.state_forward = True

        # Узнаем, у кого сейчас обратное распространение по инвертированному флагу.
        if not self.generator_training:
            return self.generator.backward_graph_batch()
        else:
            return self.discriminator.backward_graph_batch()
    
