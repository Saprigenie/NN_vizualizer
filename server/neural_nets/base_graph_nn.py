import torch.nn as nn

from utility import create_batch


class BaseGraphNN(nn.Module):
    def __init__(self, in_features, out_features, batch_size, lr, name, dataset_i):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        self.loss_value = 0

        # Слои нейронной сети, которая принимает in_features параметров и
        # выдает out_features параметров.
        # ----- Структура сети -------
        pass
        # ----------------------------

        # Также необходимо определить размер батча обучения нейронной сети, на какой мы эпохе и 
        # на каком индексе в тренировочном датасете мы остановились
        # для удобства.
        self.batch_size = batch_size
        self.lr = lr
        self.curr_epoch = 1
        self.train_i = 0
        self.dataset_i = dataset_i

    def set_batch_size(self, batch_size):
        """
        Устанавливает размер батча обучения нейронной сети.
        """
        self.batch_size = batch_size

    def set_lr(self, lr):
        """
        Устанавливает размер батча обучения нейронной сети.
        """
        self.lr = lr
        if self.optimizer:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    def set_optimizer(self, optimizer, lr):
        """
        Задает оптимизатор, который будет использовтаь нейронная сеть:
        """
        self.optimizer = optimizer(self.parameters(), lr)

    def set_loss_function(self, loss_function):
        """
        Задает функцию потерь, которую будет использовтаь нейронная сеть:
        """
        self.loss_function = loss_function()

    def forward(self, x):
        """
        Реализация прямого прохода по слоям нейронной сети.
        """
        pass

    def train_batch(self, train_dataset):
        """
        1 батч обучения нейронной сети.
        """
        pass

    def graph_structure():
        """
        Возвращает список словарей, в котором разбита структура сети по слоям.
        При изменении структуры здесь, необходимо переработать индексы в
        forward_graph и backward_graph
        Например:
        {
            "model": self.name,
            "loss": self.loss_value,
            "structure": [
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
        }
        """
        pass

    def forward_graph(self, x):
        """
        Реализация прямого прохода по слоям нейронной сети с возвратом частично преобразованных данных, чтобы была
        возможность отображения в слоях "Data" и "DataImage" в графовом представлении на клиенте.
        """
        pass

    def form_train_step(self, dataset_len, train_i):
        """
        Формирует на каком мы наборе данных, на каком батче и на какой эпохе.
        """
        return { 
            "data": {"curr": 1, "max": self.batch_size},
            "batch": {"curr": (train_i // self.batch_size), "max": dataset_len // self.batch_size},
            "epoch": {"curr": self.curr_epoch},
        }

    def form_train_state(self, start_type, forward_weights, backward_weights, dataset_len, train_i = None):
        """
        Возвращает стандартное сосотояние нейронной сети для отправки на клиент.
        """
        if (train_i is None): train_i = self.train_i
        return {
            "model": self.name,
            "loss": self.loss_value,
            "type": start_type,
            "ended": False,
            "forwardWeights": forward_weights,
            "backwardWeights": backward_weights,
            "trainStep": self.form_train_step(dataset_len, train_i)
        }

    def forward_graph_batch(self, train_dataset, start_index = None):
        """
        forward_graph для целого батча.
        """
        if (start_index is None): start_index = self.train_i
        x_batch, _ = create_batch(train_dataset, start_index, self.batch_size)
        return {
            "dataIndex": 0,
            "layerIndex": 0,
            "weights": [self.forward_graph(data) for data in x_batch]
        }

    def backward_graph_batch(self):
        """
        Проходимся по всем слоям, которые меняют свои веса в ходе
        обратного распространения ошибки и добаляем их вместе с их
        номерами в графовом представлении нейронной сети graph_structure()
        """
        pass

    def graph_batch(self, train_dataset):
        """
        Прямой и обратный проход по нейронной сети.
        """
        forward_weights = self.forward_graph_batch(train_dataset)
        self.train_batch(train_dataset)
        backward_weights = self.backward_graph_batch(train_dataset)

        return self.form_train_state("forward", forward_weights, backward_weights, len(train_dataset))

