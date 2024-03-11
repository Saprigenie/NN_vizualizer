from flask_restx import Namespace, Resource
from flask import abort, session


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