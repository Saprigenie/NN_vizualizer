from flask_restx import Namespace, Resource
from flask import abort, session

from config import NN_NAMES
from neural_nets.ann import ANN
from neural_nets.cnn import CNN
from neural_nets.gan import GAN


api = Namespace("nn", description="Операции с нейронными сетями.")

@api.route('/state/<nn_name>')
class NNStates(Resource):
    def get(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        structure = model.graph_structure()
        return structure
    
@api.route('/train/<nn_name>')
class NNTrain(Resource):
    def get(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name) 
        
        if model.state_forward:
            weights_update = model.forward_graph_batch(session.get("digits_dataset"))
        else:
            model.train_batch(session.get("digits_dataset"))
            weights_update = model.backward_graph_batch()


        return weights_update
    
@api.route('/restart/<nn_name>')
class BatchSize(Resource):
    def put(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name) 
        
        # Сохраняем некоторые параметры, которые не хотим потерять.
        batch_size = model.batch_size

        if NN_NAMES[0] == nn_name:
            session[nn_name] = ANN()
        elif NN_NAMES[1] == nn_name:
            session[nn_name] = CNN()
        elif NN_NAMES[2] == nn_name:
            session[nn_name] = GAN()

        # Устанавливаем их заново.
        model = session.get(nn_name)
        model.set_batch_size(batch_size)

        return 'Ok'
    

@api.route('/batch_size/<nn_name>/<batch_size>')
class BatchSize(Resource):
    def put(self, nn_name, batch_size):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name) 
        
        model.set_batch_size(int(batch_size))

        return 'Ok'