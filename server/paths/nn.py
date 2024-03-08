from flask_restx import Namespace, Resource
from flask import abort, session

from config import NN_NAMES
from neural_nets.ann import ANN
from neural_nets.cnn import CNN
from neural_nets.gan import GAN


api = Namespace("nn", description="Операции с нейронными сетями.")

@api.route('/state/<nn_name>')
class ImagesList(Resource):
    def get(self, nn_name):
        if not session.get(nn_name):
            if nn_name == NN_NAMES[0]:
                session["model"] = ANN()
            elif nn_name == NN_NAMES[1]:
                session["model"] = CNN()
            elif nn_name == NN_NAMES[2]:
                session["model"] = GAN()
            else:
                abort(404)


        model = session.get("model") 
        structure = model.graph_structure()
        return structure