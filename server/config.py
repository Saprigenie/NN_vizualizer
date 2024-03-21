from torch import cuda
from os import environ

NN_NAMES = ['ann', 'cnn', 'gan']
FLASK_PORT = environ.get('FLASK_PORT', 5000)
SESSION_PERMANENT = False
SESSION_TYPE = "filesystem"