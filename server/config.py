from torch import cuda
from os import environ

NN_NAMES = ['ann', 'cnn', 'gan']
FLASK_PORT = environ.get('FLASK_PORT', 5000)
DEVICE = "cuda" if cuda.is_available() else "cpu"