from flask import session

from config import NN_NAMES
from neural_nets.small_ann import SmallANN
from neural_nets.ann import ANN
from neural_nets.cnn import CNN
from neural_nets.gan import GAN


def init_session():
    if not session.get("initialized"):
        session["initialized"] = True

        # Инициализируем нейронные сети.
        session[NN_NAMES[0]] = SmallANN()
        session[NN_NAMES[1]] = ANN()
        session[NN_NAMES[2]] = CNN()
        session[NN_NAMES[3]] = GAN()
