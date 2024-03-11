import { STANDART_GRAPH_GAP } from '@/constants'

export function drawGraph(cy, graphData, offset = 0, idPrefix = '') {
  // Добавляем сами узлы графа отображения нейронной сети.
  for (let layerNum = 0; layerNum < graphData.length; layerNum++) {
    let layer = graphData[layerNum]
    switch (layer.type) {
      case 'Data':
        offset = addData(cy, layer, layerNum, offset, idPrefix)
        break
      case 'DataImage':
        offset = addDataImage(cy, layer, layerNum, offset, idPrefix)
        break
      case 'Linear':
        offset = addLinear(cy, layer, layerNum, offset, idPrefix)
        break
      case 'Activation':
        offset = addActivation(cy, layer, layerNum, offset, idPrefix)
        break
      case 'Conv2d':
        offset = addConv2d(cy, layer, layerNum, offset, idPrefix)
        break
      case 'MaxPool2d':
        offset = addMaxPool2d(cy, layer, layerNum, offset, idPrefix)
        break
      case 'MergeFlatten':
        offset = addMergeFlatten(cy, layer, layerNum, offset, idPrefix)
        break
    }
  }
  // Добавляем связи.
  for (let layerNum = 0; layerNum < graphData.length; layerNum++) {
    let layer = graphData[layerNum]
    if ((layer.type = 'Connection')) {
      addConnection(cy, layer, layerNum, idPrefix)
    }
  }

  // Возвращаем занятое нейронной сетью пространство.
  return offset
}

function addData(cy, layer, layerNum, offset = 0, idPrefix = '', nodeSize = [50, 50], spacing = 0) {
  let centerCoeff = (layer.count * nodeSize[1] + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    cy.add({
      group: 'nodes',
      data: {
        id: idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
        type: 'Data',
        value: layer.weights[nodeNum]
      },
      position: {
        x: offset + nodeSize[0] / 2,
        y: (nodeSize[1] + spacing) * nodeNum + nodeSize[0] / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: nodeSize[0] + 'px',
        height: nodeSize[1] + 'px'
      },
      locked: true,
      classes: ['data']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize[1] + STANDART_GRAPH_GAP
}

function addDataImage(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = [50, 50],
  spacing = 50
) {
  let centerCoeff =
    (layer.count[0] * layer.count[2] * nodeSize[1] + spacing * (layer.count[0] - 1)) / 2
  for (let imageNum = 0; imageNum < layer.count[0]; imageNum++) {
    // Добавляем 1 node - который в себе будет содержать данные изображения.
    let elementSize = [layer.count[1] * nodeSize[0], layer.count[2] * nodeSize[1]]
    cy.add({
      group: 'nodes',
      data: {
        id: idPrefix + '_' + layerNum + '_' + imageNum + 'N',
        type: 'DataImage',
        value: 'Data'
      },
      position: {
        x: offset + elementSize[0] / 2,
        y:
          imageNum * layer.count[2] * nodeSize[1] + // offset по количеству пикселей в столбце.
          imageNum * spacing +
          elementSize[1] / 2 -
          centerCoeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: elementSize[0] + 'px',
        height: elementSize[1] + 'px'
      },
      locked: true,
      classes: ['data', 'dataImage']
    })

    for (let h = 0; h < layer.count[1]; h++) {
      for (let w = 0; w < layer.count[2]; w++) {
        // Добавляем сами данные.
        cy.add({
          group: 'nodes',
          data: {
            id: idPrefix + '_image_' + imageNum + '_' + layerNum + '_' + h + '_' + w + 'N',
            type: 'DataImage',
            value: layer.weights[imageNum][h][w]
          },
          position: {
            x: offset + w * nodeSize[0] + nodeSize[0] / 2,
            y:
              imageNum * layer.count[2] * nodeSize[1] + // offset разных изображений в слое.
              imageNum * spacing +
              h * nodeSize[1] + // offset по номеру пикселя в строке.
              nodeSize[1] / 2 -
              centerCoeff // Центруем относительно начального угла обзора.
          },
          style: {
            width: nodeSize[0] + 'px',
            height: nodeSize[1] + 'px'
          },
          locked: true,
          classes: ['data']
        })
      }
    }
  }
  // Возвращает offset для следующего слоя.
  return offset + layer.count[1] * nodeSize[1] + STANDART_GRAPH_GAP
}

function addLinear(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = [100, 40],
  spacing = 50
) {
  let centerCoeff = (layer.count * nodeSize[1] + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    cy.add({
      group: 'nodes',
      data: {
        id: idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
        type: 'Linear',
        value: 'Linear\nbias:' + Number.parseFloat(layer.bias[nodeNum]).toFixed(4)
      },
      position: {
        x: offset + nodeSize[0] / 2,
        y: (nodeSize[1] + spacing) * nodeNum + nodeSize[1] / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: nodeSize[0] + 'px',
        height: nodeSize[1] + 'px'
      },
      locked: true,
      classes: ['linear']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize[1] + STANDART_GRAPH_GAP
}

function addActivation(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = [100, 40],
  spacing = 50
) {
  let centerCoeff = (layer.count * nodeSize[1] + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    cy.add({
      group: 'nodes',
      data: {
        id: idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
        type: 'Activation',
        value: 'Activation\ntype: ' + layer.activation
      },
      position: {
        x: offset + nodeSize[0] / 2,
        y: (nodeSize[1] + spacing) * nodeNum + nodeSize[1] / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: nodeSize[0] + 'px',
        height: nodeSize[1] + 'px'
      },
      locked: true,
      classes: ['activation']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize[1] + STANDART_GRAPH_GAP
}

function addConv2d(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = [50, 50],
  spacing = 150
) {
  let centerCoeff =
    (layer.count[0] * layer.count[2] * nodeSize[1] + spacing * (layer.count[0] - 1)) / 2
  for (let convNum = 0; convNum < layer.count[0]; convNum++) {
    // Добавляем 1 node - который в себе будет содержать данные всего Conv2d.
    let elementSize = [layer.count[1] * nodeSize[0], layer.count[2] * nodeSize[1]]
    let constValues = 'Conv2d:' + '\npadding: ' + layer.padding + '\nstride: ' + layer.stride
    cy.add({
      group: 'nodes',
      data: {
        id: idPrefix + '_' + layerNum + '_' + convNum + 'N',
        type: 'Conv2d',
        constValues: constValues,
        value: constValues + '\nbias: ' + Number.parseFloat(layer.bias[convNum]).toFixed(4)
      },
      position: {
        x: offset + elementSize[0] / 2,
        y:
          convNum * layer.count[2] * nodeSize[1] + // offset по количеству пикселей в столбце.
          convNum * spacing +
          elementSize[1] / 2 -
          centerCoeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: elementSize[0] + nodeSize[0] / 2 + 'px',
        height: elementSize[1] + nodeSize[1] / 2 + 'px'
      },
      locked: true,
      classes: ['convolution']
    })

    for (let h = 0; h < layer.count[1]; h++) {
      for (let w = 0; w < layer.count[2]; w++) {
        // Добавляем данные фильтра.
        cy.add({
          group: 'nodes',
          data: {
            id: idPrefix + '_image_' + convNum + '_' + layerNum + '_' + h + '_' + w + 'N',
            type: 'DataImage',
            value: Number.parseFloat(layer.weights[convNum][h][w]).toFixed(3)
          },
          position: {
            x: offset + w * nodeSize[0] + nodeSize[0] / 2,
            y:
              convNum * layer.count[2] * nodeSize[1] + // offset разных изображений в слое.
              convNum * spacing +
              h * nodeSize[1] + // offset по номеру пикселя в строке.
              nodeSize[1] / 2 -
              centerCoeff // Центруем относительно начального угла обзора.
          },
          style: {
            width: nodeSize[0] + 'px',
            height: nodeSize[1] + 'px'
          },
          locked: true,
          classes: ['data']
        })
      }
    }
  }
  // Возвращает offset для следующего слоя.
  return offset + layer.count[1] * nodeSize[1] + STANDART_GRAPH_GAP
}

function addMaxPool2d(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = [100, 80],
  spacing = 50
) {
  let centerCoeff = (layer.count * nodeSize[1] + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    cy.add({
      group: 'nodes',
      data: {
        id: idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
        type: 'MaxPool2d',
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
        x: offset + nodeSize[0] / 2,
        y: (nodeSize[1] + spacing) * nodeNum + nodeSize[1] / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      style: {
        width: nodeSize[0] + 'px',
        height: nodeSize[1] + 'px'
      },
      locked: true,
      classes: ['activation']
    })
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize[1] + STANDART_GRAPH_GAP
}

function addMergeFlatten(cy, layer, layerNum, offset = 0, idPrefix = '', nodeSize = [100, 20]) {
  cy.add({
    group: 'nodes',
    data: {
      id: idPrefix + '_' + layerNum + '_' + 0 + 'N',
      type: 'MergeFlatten',
      value: 'Merge Flatten'
    },
    position: {
      x: offset + nodeSize[0] / 2,
      y: nodeSize[1] + nodeSize[1] / 2
    },
    style: {
      width: nodeSize[0] + 'px',
      height: nodeSize[1] + 'px'
    },
    locked: true,
    classes: ['activation']
  })

  // Возвращает offset для следующего слоя.
  return offset + nodeSize[1] + STANDART_GRAPH_GAP
}

function addConnection(cy, connection, connectionNum, idPrefix = '') {
  // Получаем списки узлов, которые нужно соединить с помощью слоя связей.
  let sources = cy.filter(function (element, i) {
    let id = element.id()
    return element.isNode() && id.startsWith(idPrefix + '_' + (connectionNum - 1) + '_')
  })
  let targets = cy.filter(function (element, i) {
    let id = element.id()
    return element.isNode() && id.startsWith(idPrefix + '_' + (connectionNum + 1) + '_')
  })

  for (let targNum = 0; targNum < targets.length; targNum++) {
    if (Array.isArray(connection.weights[targNum])) {
      // Значит каждый узел прошлого слоя связан с каждым узлом следующего.
      for (let sourceNum = 0; sourceNum < sources.length; sourceNum++) {
        addEdge(
          cy,
          idPrefix + '_' + connectionNum + '_' + sourceNum + '_' + targNum + 'E',
          sources[sourceNum].id(),
          targets[targNum].id(),
          Number.parseFloat(connection.weights[targNum][sourceNum]).toFixed(4)
        )
      }
    } else {
      // Значит каждый узел предыдущего слоя связан с 1 узлом следующего слоя.
      addEdge(
        cy,
        idPrefix + '_' + connectionNum + '_' + targNum + '_' + targNum + 'E',
        sources[targNum].id(),
        targets[targNum].id()
      )
    }
  }
}

function addEdge(cy, id, source, target, weight = NaN) {
  let edgeParameters = {
    group: 'edges',
    data: {
      type: 'Connection',
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
