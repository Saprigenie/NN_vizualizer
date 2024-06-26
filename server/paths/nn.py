from flask_restx import Namespace, Resource
from flask import abort, session, send_file, request
import torch
import io

from config import NN_NAMES
from neural_nets.small_ann import SmallANN
from neural_nets.ann import ANN
from neural_nets.cnn import CNN
from neural_nets.gan import GAN
from datasets.load_dataset import load_digits_dataset, load_xor_dataset


api = Namespace("nn", description="Операции с нейронными сетями.")
datasets = [load_xor_dataset(), load_digits_dataset()]


@api.route("/state/<nn_name>")
class NNStates(Resource):
    def get(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        structure = model.graph_structure()
        return structure


@api.route("/train/<nn_name>")
class NNTrain(Resource):
    def get(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        return model.graph_batch(datasets[model.dataset_i])


@api.route("/restart/<nn_name>")
class NNRestart(Resource):
    def put(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        # Сохраняем некоторые параметры, которые не хотим потерять.
        batch_size = model.batch_size

        if NN_NAMES[0] == nn_name:
            session[nn_name] = SmallANN()
        elif NN_NAMES[1] == nn_name:
            session[nn_name] = ANN()
        elif NN_NAMES[2] == nn_name:
            session[nn_name] = CNN()
        elif NN_NAMES[3] == nn_name:
            session[nn_name] = GAN()

        # Устанавливаем их заново.
        model = session.get(nn_name)
        model.set_batch_size(batch_size)

        return "Ok"


@api.route("/batch_size/<nn_name>", defaults={"batch_size": 5})
@api.route("/batch_size/<nn_name>/<batch_size>")
class BatchSize(Resource):
    def get(self, nn_name, batch_size):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        return model.batch_size

    def put(self, nn_name, batch_size):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        model.set_batch_size(int(batch_size))

        return "Ok"


@api.route("/learn_rate/<nn_name>", defaults={"lr": 0.05})
@api.route("/learn_rate/<nn_name>/<lr>")
class LearnRate(Resource):
    def get(self, nn_name, lr):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        return model.lr

    def put(self, nn_name, lr):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        model.set_lr(float(lr))

        return "Ok"


@api.route("/weights/<nn_name>")
class Weights(Resource):
    def get(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        weights_buffer = io.BytesIO()
        torch.save(model.state_dict(), weights_buffer)

        return send_file(
            io.BytesIO(weights_buffer.getbuffer()),
            download_name=model.name + ".pth",
            as_attachment=True,
        )

    def post(self, nn_name):
        if not session.get(nn_name):
            abort(404)
        else:
            model = session.get(nn_name)

        weights = request.files["weights"]

        # TO DO: Ловля ошибок, потенциально небезопасная штучка.
        model.load_state_dict(torch.load(weights))
        model.eval()

        return "Ok"
