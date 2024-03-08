from flask_restx import Api

from .nn import api as nn_api

api = Api(
    title="NN training visualizer",
    version="1.0",
    prefix='/api'
)

api.add_namespace(nn_api)