from flask import session

from config import NN_NAMES
from neural_nets.ann import ANN
from neural_nets.cnn import CNN
from neural_nets.gan import GAN

from datasets.load_dataset import load_digits_dataset


def init_session():
    if not session.get("initialized"):
        session["initialized"] = True

        # Инициализируем нейронные сети.
        session[NN_NAMES[0]] = ANN()
        session[NN_NAMES[1]] = CNN()
        session[NN_NAMES[2]] = GAN()

        # Загружаем тренировочные данные в сессию
        session["digits_dataset"] = load_digits_dataset()
        

        
