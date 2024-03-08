from flask_restx import Namespace, Resource
from flask import abort

from config import NN_NAMES
from neural_nets.ann import ANN
from neural_nets.cnn import CNN
from neural_nets.gan import GAN


api = Namespace("nn", description="Операции с нейронными сетями.")

@api.route('/state/<nn_name>')
class ImagesList(Resource):
    def get(self, nn_name):
        # TO DO: добавить нормальные сессии, чтобы запоминалось создание нейронки для того или иного пользователя
        # а не пересоздавалось каждый раз.
        if nn_name == NN_NAMES[0]:
            model = ANN()
        elif nn_name == NN_NAMES[1]:
            model = CNN()
        elif nn_name == NN_NAMES[2]:
            model = GAN()
        else:
            abort(404)

    
        structure = model.graph_structure()
        return structure