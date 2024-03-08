from flask_restx import Namespace, Resource
from flask import abort, session


api = Namespace("nn", description="Операции с нейронными сетями.")

@api.route('/state/<nn_name>')
class ImagesList(Resource):
    def get(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        structure = model.graph_structure()
        return structure