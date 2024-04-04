from datetime import timedelta
from os import environ

NN_NAMES = ['smallann', 'ann', 'cnn', 'gan']
FLASK_PORT = environ.get('FLASK_PORT', 5000)
SESSION_TYPE = "filesystem"
PERMANENT_SESSION_LIFETIME = timedelta(days = 1)