import torch.nn as nn


class BaseGraphNN(nn.Module):
    def __init__(self, in_features, out_features, batch_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Слои нейронной сети, которая принимает in_features параметров и
        # выдает out_features параметров.
        pass

        # Также необходимо определить размер батча обучения нейронной сети и 
        # на каком индексе в тренировочном датасете мы остановились
        # для удобства.
        self.batch_size = batch_size
        self.forward_i = 0
        self.train_i = 0

        # Какое состояние нейросети для отправки весов на клиент (forward или backward)
        self.state_forward = True 

    def set_batch_size(self, batch_size):
        """
        Устанавливает размер батча обучения нейронной сети.
        """
        self.batch_size = batch_size

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

    def forward_graph(self, x):
        """
        Реализация прямого прохода по слоям нейронной сети с возвратом частично преобразованных данных, чтобы была
        возможность отображения в слоях "Data" и "DataImage" в графовом представлении на клиенте.
        """
        pass

    def forward_graph_batch(self, train_dataset):
        """
        forward_graph для целого батча.
        """
        pass

    def backward_graph_batch(self):
        """
        Проходимся по всем слоям, которые меняют свои веса в ходе
        обратного распространения ошибки и добаляем их вместе с их
        номерами в графовом представлении нейронной сети graph_structure()
        """
        pass
