import { STANDART_GRAPH_GAP } from '@/constants'

export function drawGraph(cy, graphData, offset = 0, id_prefix = '') {
  // Добавляем сами узлы графа отображения нейронной сети.
  for (let layer_num = 0; layer_num < graphData.length; layer_num++) {
    let layer = graphData[layer_num]
    switch (layer.type) {
      case 'Data':
        offset = addData(cy, layer, layer_num, offset, id_prefix)
        break
      case 'DataImage':
        offset = addDataImage(cy, layer, layer_num, offset, id_prefix)
        break
      case 'Linear':
        offset = addLinear(cy, layer, layer_num, offset, id_prefix)
        break
      case 'Activation':
        offset = addActivation(cy, layer, layer_num, offset, id_prefix)
        break
      case 'Conv2d':
        offset = addConv2d(cy, layer, layer_num, offset, id_prefix)
        break
      case 'MaxPool2d':
        offset = addMaxPool2d(cy, layer, layer_num, offset, id_prefix)
        break
      case 'MergeFlatten':
        offset = addMergeFlatten(cy, layer, layer_num, offset, id_prefix)
        break
    }
  }
  // Добавляем связи.
  for (let layer_num = 0; layer_num < graphData.length; layer_num++) {
    let layer = graphData[layer_num]
    if ((layer.type = 'Connection')) {
      addConnection(cy, layer, layer_num, id_prefix)
    }
  }
}

function addData(
  cy,
  layer,
  layer_num,
  offset = 0,
  id_prefix = '',
  node_size = [50, 50],
  spacing = 0
) {
  let center_coeff = (layer.count * node_size[1] + spacing * (layer.count - 1)) / 2
  for (let node_num = 0; node_num < layer.count; node_num++) {
    cy.add({
      group: 'nodes',
      data: {
        id: id_prefix + layer_num + '_' + node_num + '_n',
        type: 'data',
        value: layer.weights[node_num]
      },
      position: {
        x: offset + node_size[0] / 2,
        y: (node_size[1] + spacing) * node_num + node_size[0] / 2 - center_coeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: node_size[0] + 'px',
        height: node_size[1] + 'px'
      },
      locked: true,
      classes: ['data']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + node_size[1] + STANDART_GRAPH_GAP
}

function addDataImage(
  cy,
  layer,
  layer_num,
  offset = 0,
  id_prefix = '',
  node_size = [50, 50],
  spacing = 50
) {
  let center_coeff =
    (layer.count[0] * layer.count[2] * node_size[1] + spacing * (layer.count[0] - 1)) / 2
  for (let image_num = 0; image_num < layer.count[0]; image_num++) {
    // Добавляем 1 node - который в себе будет содержать данные изображения.
    let element_size = [layer.count[1] * node_size[0], layer.count[2] * node_size[1]]
    cy.add({
      group: 'nodes',
      data: {
        id: id_prefix + layer_num + '_' + image_num + '_n',
        type: 'data',
        value: 'Data'
      },
      position: {
        x: offset + element_size[0] / 2,
        y:
          image_num * layer.count[2] * node_size[1] + // offset по количеству пикселей в столбце.
          image_num * spacing +
          element_size[1] / 2 -
          center_coeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: element_size[0] + 'px',
        height: element_size[1] + 'px'
      },
      locked: true,
      classes: ['data', 'dataImage']
    })

    for (let w = 0; w < layer.count[1]; w++) {
      for (let h = 0; h < layer.count[2]; h++) {
        // Добавляем сами данные.
        cy.add({
          group: 'nodes',
          data: {
            id: id_prefix + 'image_' + image_num + '_' + layer_num + '_' + w + '_' + h + '_n',
            type: 'dataImage',
            value: layer.weights[image_num][h][w]
          },
          position: {
            x: offset + w * node_size[0] + node_size[0] / 2,
            y:
              image_num * layer.count[2] * node_size[1] + // offset разных изображений в слое.
              image_num * spacing +
              h * node_size[1] + // offset по номеру пикселя в строке.
              node_size[1] / 2 -
              center_coeff // Центруем относительно начального угла обзора.
          },
          style: {
            width: node_size[0] + 'px',
            height: node_size[1] + 'px'
          },
          locked: true,
          classes: ['data']
        })
      }
    }
  }
  // Возвращает offset для следующего слоя.
  return offset + layer.count[1] * node_size[1] + STANDART_GRAPH_GAP
}

function addLinear(
  cy,
  layer,
  layer_num,
  offset = 0,
  id_prefix = '',
  node_size = [100, 40],
  spacing = 50
) {
  let center_coeff = (layer.count * node_size[1] + spacing * (layer.count - 1)) / 2
  for (let node_num = 0; node_num < layer.count; node_num++) {
    cy.add({
      group: 'nodes',
      data: {
        id: id_prefix + layer_num + '_' + node_num + '_n',
        type: 'linear',
        value: 'Linear\nbias:' + Number.parseFloat(layer.bias[node_num]).toFixed(4)
      },
      position: {
        x: offset + node_size[0] / 2,
        y: (node_size[1] + spacing) * node_num + node_size[1] / 2 - center_coeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: node_size[0] + 'px',
        height: node_size[1] + 'px'
      },
      locked: true,
      classes: ['linear']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + node_size[1] + STANDART_GRAPH_GAP
}

function addActivation(
  cy,
  layer,
  layer_num,
  offset = 0,
  id_prefix = '',
  node_size = [100, 40],
  spacing = 50
) {
  let center_coeff = (layer.count * node_size[1] + spacing * (layer.count - 1)) / 2
  for (let node_num = 0; node_num < layer.count; node_num++) {
    cy.add({
      group: 'nodes',
      data: {
        id: id_prefix + layer_num + '_' + node_num + '_n',
        type: 'activation',
        value: 'Activation\ntype: ' + layer.activation
      },
      position: {
        x: offset + node_size[0] / 2,
        y: (node_size[1] + spacing) * node_num + node_size[1] / 2 - center_coeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: node_size[0] + 'px',
        height: node_size[1] + 'px'
      },
      locked: true,
      classes: ['activation']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + node_size[1] + STANDART_GRAPH_GAP
}

function addConv2d(
  cy,
  layer,
  layer_num,
  offset = 0,
  id_prefix = '',
  node_size = [50, 50],
  spacing = 150
) {
  let center_coeff =
    (layer.count[0] * layer.count[2] * node_size[1] + spacing * (layer.count[0] - 1)) / 2
  for (let conv_num = 0; conv_num < layer.count[0]; conv_num++) {
    // Добавляем 1 node - который в себе будет содержать данные всего Conv2d.
    let element_size = [layer.count[1] * node_size[0], layer.count[2] * node_size[1]]
    cy.add({
      group: 'nodes',
      data: {
        id: id_prefix + layer_num + '_' + conv_num + '_n',
        type: 'conv2d',
        value:
          'Conv2d:' +
          '\npadding: ' +
          layer.padding +
          '\nstride: ' +
          layer.stride +
          '\nbias: ' +
          Number.parseFloat(layer.bias[conv_num]).toFixed(4)
      },
      position: {
        x: offset + element_size[0] / 2,
        y:
          conv_num * layer.count[2] * node_size[1] + // offset по количеству пикселей в столбце.
          conv_num * spacing +
          element_size[1] / 2 -
          center_coeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: element_size[0] + node_size[0] / 2 + 'px',
        height: element_size[1] + node_size[1] / 2 + 'px'
      },
      locked: true,
      classes: ['convolution']
    })

    for (let w = 0; w < layer.count[1]; w++) {
      for (let h = 0; h < layer.count[2]; h++) {
        // Добавляем данные фильтра.
        cy.add({
          group: 'nodes',
          data: {
            id: id_prefix + 'image_' + conv_num + '_' + layer_num + '_' + w + '_' + h + '_n',
            type: 'dataImage',
            value: Number.parseFloat(layer.weights[conv_num][h][w]).toFixed(3)
          },
          position: {
            x: offset + w * node_size[0] + node_size[0] / 2,
            y:
              conv_num * layer.count[2] * node_size[1] + // offset разных изображений в слое.
              conv_num * spacing +
              h * node_size[1] + // offset по номеру пикселя в строке.
              node_size[1] / 2 -
              center_coeff // Центруем относительно начального угла обзора.
          },
          style: {
            width: node_size[0] + 'px',
            height: node_size[1] + 'px'
          },
          locked: true,
          classes: ['data']
        })
      }
    }
  }
  // Возвращает offset для следующего слоя.
  return offset + layer.count[1] * node_size[1] + STANDART_GRAPH_GAP
}

function addMaxPool2d(
  cy,
  layer,
  layer_num,
  offset = 0,
  id_prefix = '',
  node_size = [100, 80],
  spacing = 50
) {
  let center_coeff = (layer.count * node_size[1] + spacing * (layer.count - 1)) / 2
  for (let node_num = 0; node_num < layer.count; node_num++) {
    cy.add({
      group: 'nodes',
      data: {
        id: id_prefix + layer_num + '_' + node_num + '_n',
        type: 'maxpool2d',
        value:
          'MaxPool2d' +
          '\nkernel size: ' +
          layer.kernelSize +
          '\npadding: ' +
          layer.padding +
          '\nstride: ' +
          layer.stride
      },
      position: {
        x: offset + node_size[0] / 2,
        y: (node_size[1] + spacing) * node_num + node_size[1] / 2 - center_coeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: node_size[0] + 'px',
        height: node_size[1] + 'px'
      },
      locked: true,
      classes: ['activation']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + node_size[1] + STANDART_GRAPH_GAP
}

function addMergeFlatten(cy, layer, layer_num, offset = 0, id_prefix = '', node_size = [100, 20]) {
  cy.add({
    group: 'nodes',
    data: {
      id: id_prefix + layer_num + '_' + 0 + '_n',
      type: 'mergeflatten',
      value: 'Merge Flatten'
    },
    position: {
      x: offset + node_size[0] / 2,
      y: node_size[1] + node_size[1] / 2
    },
    style: {
      width: node_size[0] + 'px',
      height: node_size[1] + 'px'
    },
    locked: true,
    classes: ['activation']
  })

  // Возвращает offset для следующего слоя.
  return offset + node_size[1] + STANDART_GRAPH_GAP
}

function addConnection(cy, connection, connection_num, id_prefix = '') {
  // Получаем списки узлов, которые нужно соединить с помощью слоя связей.
  let sources = cy.filter(function (element, i) {
    let id = element.data('id')
    return element.isNode() && id.startsWith(id_prefix + (connection_num - 1) + '_')
  })
  let targets = cy.filter(function (element, i) {
    let id = element.data('id')
    return element.isNode() && id.startsWith(id_prefix + (connection_num + 1) + '_')
  })

  for (let targ_num = 0; targ_num < targets.length; targ_num++) {
    if (Array.isArray(connection.weights[targ_num])) {
      // Значит каждый узел прошлого слоя связан с каждым узлом следующего.
      for (let source_num = 0; source_num < sources.length; source_num++) {
        addEdge(
          cy,
          id_prefix + connection_num + '_' + source_num + '_' + targ_num + '_e',
          sources[source_num].data('id'),
          targets[targ_num].data('id'),
          Number.parseFloat(connection.weights[targ_num][source_num]).toFixed(4)
        )
      }
    } else {
      // Значит каждый узел предыдущего слоя связан с 1 узлом следующего слоя.
      addEdge(
        cy,
        id_prefix + connection_num + '_' + targ_num + '_' + targ_num + '_e',
        sources[targ_num].data('id'),
        targets[targ_num].data('id')
      )
    }
  }
}

function addEdge(cy, id, source, target, weight = NaN) {
  let edgeParameters = {
    group: 'edges',
    data: {
      id: id,
      source: source,
      target: target
    },
    locked: true,
    classes: []
  }

  if (!isNaN(weight)) {
    edgeParameters.data.value = weight
    edgeParameters.classes.push('ehasweights')
  } else {
    edgeParameters.classes.push('enothasweights')
  }

  // Добавляем сформированную связь в граф.
  cy.add(edgeParameters)
}
