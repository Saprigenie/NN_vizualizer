import torch


def get_layers(model: torch.nn.Module):
    children = list(model.children())
    flatt_children = []
    if children == []:
        # Если у модели нет слоев, то она сама слой.
        return [model]
    else:
       # Итерируемся по слоям, ища вложенные.
       for child in children:
            flatt_children.extend(get_layers(child))
    return flatt_children


def graph_rep_add_data(count, weights):
    structure = []
    structure.append({
        "type": "Data",
        "count": count,
        "weights": weights
    })
    
    return structure


def graph_rep_add_image_data(count, weights):
    structure = []
    structure.append({
        "type": "DataImage",
        "count": count,
        "weights": weights
    })
    
    return structure


def graph_rep_add_connection(weights, display_weights = True):
    structure = []

    structure.append({
        "type": "Connection",
        "displayWeights": display_weights,
        "weights": weights
    })

    return structure 


def graph_rep_add_linear(layer: torch.nn.Linear, activation_type = "ReLU"):
    structure = []

    structure.append({
        "type": "Linear",
        "count": layer.weight.shape[0],
        "bias": layer.bias.tolist()
    })

    structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

    structure.append({
        "type": "Activation",
        "count": layer.weight.shape[0],
        "activation": activation_type
    })

    return structure


def graph_rep_add_conv2d(layer: torch.nn.Linear, activation_type = "ReLU"):
    structure = []

    structure.append({
        "type": "Conv2d",
        "count": [layer.weight.shape[0], layer.kernel_size[0], layer.kernel_size[0]],
        "stride": layer.stride[0],
        "padding": layer.padding[0],
        "bias": layer.bias.tolist(),
        "weights": layer.weight.reshape([-1, layer.kernel_size[0], layer.kernel_size[0]]).tolist()
    })

    structure.extend(graph_rep_add_connection([0] * layer.weight.shape[0], False))

    structure.append({
        "type": "Activation",
        "count": layer.weight.shape[0],
        "activation": activation_type
    })

    return structure


def graph_rep_add_maxpool2d(layer: torch.nn.Linear, count):
    structure = []

    structure.append({
        "type": "MaxPool2d",
        "count": count,
        "kernelSize": layer.kernel_size,
        "stride": layer.stride,
        "padding": layer.padding
    })

    return structure


def graph_rep_add_flatten():
    structure = []

    structure.append({
        "type": "MergeFlatten"
    })

    return structure