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

  // Добавляем рамку с названием нейронной сети и loss-ом.
  // addNNframe(cy, graphData.model, graphData.loss, [offset, maxOffsetY], idPrefix)

  // Возвращаем занятое нейронной сетью пространство.
  return offset
}

function addData(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = { x: 50, y: 50 },
  spacing = 0
) {
  let centerCoeff = (layer.count * nodeSize.y + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    addNode(
      cy,
      idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
      'Data',
      layer.weights[nodeNum],
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['data']
    )
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize.x + STANDART_GRAPH_GAP
}

function addDataImage(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = { x: 50, y: 50 },
  spacing = 50
) {
  let blocksN = layer.count[0]
  let cols = layer.count[2]
  let rows = layer.count[1]

  let blockSize = { x: rows * nodeSize.x, y: cols * nodeSize.y }

  let centerCoeff = (blocksN * blockSize.y + spacing * (blocksN - 1)) / 2
  for (let currBlock = 0; currBlock < blocksN; currBlock++) {
    // Добавляем 1 node - который в себе будет содержать данные изображения.
    addNode(
      cy,
      idPrefix + '_' + layerNum + '_' + currBlock + 'N',
      'DataImage',
      'Data',
      blockSize,
      {
        x: offset + blockSize.x / 2,
        y:
          currBlock * (blockSize.y + spacing) + // offset по количеству пикселей в столбце.
          blockSize.y / 2 -
          centerCoeff // Центруем относительно начального угла обзора.
      },
      ['dataImage']
    )

    for (let h = 0; h < rows; h++) {
      for (let w = 0; w < cols; w++) {
        // Добавляем сами данные.
        addNode(
          cy,
          idPrefix + '_image_' + currBlock + '_' + layerNum + '_' + h + '_' + w + 'N',
          'DataImage',
          layer.weights[currBlock][h][w],
          nodeSize,
          {
            x: offset + w * nodeSize.x + nodeSize.x / 2,
            y:
              currBlock * (blockSize.y + spacing) + // offset разных изображений в слое.
              h * nodeSize.y + // offset по номеру пикселя в строке.
              nodeSize.y / 2 -
              centerCoeff // Центруем относительно начального угла обзора.
          },
          ['data']
        )
      }
    }
  }
  // Возвращает offset для следующего слоя.
  return offset + rows * nodeSize.x + STANDART_GRAPH_GAP
}

function addLinear(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = { x: 100, y: 40 },
  spacing = 50
) {
  let centerCoeff = (layer.count * nodeSize.y + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    addNode(
      cy,
      idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
      'Linear',
      'Linear\nbias:' + Number.parseFloat(layer.bias[nodeNum]).toFixed(4),
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['linear']
    )
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize.x + STANDART_GRAPH_GAP
}

function addActivation(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = { x: 100, y: 40 },
  spacing = 50
) {
  let centerCoeff = (layer.count * nodeSize.y + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    addNode(
      cy,
      idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
      'Activation',
      'Activation\ntype: ' + layer.activation,
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['activation']
    )
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize.x + STANDART_GRAPH_GAP
}

function addConv2d(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = { x: 50, y: 50 },
  spacing = 150
) {
  let blocksN = layer.count[0]
  let cols = layer.count[2]
  let rows = layer.count[1]

  let blockSize = { x: rows * nodeSize.x, y: cols * nodeSize.y }

  let centerCoeff = (blocksN * cols * nodeSize.y + spacing * (blocksN - 1)) / 2
  let elementSize = { x: rows * nodeSize.x, y: blockSize.y }
  for (let currBlock = 0; currBlock < blocksN; currBlock++) {
    // Добавляем 1 node - который в себе будет содержать данные всего Conv2d.
    let constValues = 'Conv2d:' + '\npadding: ' + layer.padding + '\nstride: ' + layer.stride
    addNode(
      cy,
      idPrefix + '_' + layerNum + '_' + currBlock + 'N',
      'Conv2d',
      constValues + '\nbias: ' + Number.parseFloat(layer.bias[currBlock]).toFixed(4),
      {
        x: elementSize.x + nodeSize.x / 2,
        y: elementSize.y + nodeSize.y / 2
      },
      {
        x: offset + elementSize.x / 2,
        y:
          currBlock * (blockSize.y + spacing) + // offset по количеству пикселей в столбце.
          elementSize.y / 2 -
          centerCoeff // Центруем относительно начального угла обзора.
      },
      ['convolution'],
      constValues
    )

    for (let h = 0; h < rows; h++) {
      for (let w = 0; w < cols; w++) {
        // Добавляем данные фильтра.
        addNode(
          cy,
          idPrefix + '_image_' + currBlock + '_' + layerNum + '_' + h + '_' + w + 'N',
          'DataImage',
          Number.parseFloat(layer.weights[currBlock][h][w]).toFixed(3),
          nodeSize,
          {
            x: offset + w * nodeSize.x + nodeSize.x / 2,
            y:
              currBlock * (blockSize.y + spacing) + // offset разных изображений в слое.
              h * nodeSize.y + // offset по номеру пикселя в строке.
              nodeSize.y / 2 -
              centerCoeff // Центруем относительно начального угла обзора.
          },
          ['data']
        )
      }
    }
  }
  // Возвращает offset для следующего слоя.
  return offset + rows * nodeSize.x + STANDART_GRAPH_GAP
}

function addMaxPool2d(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = { x: 100, y: 80 },
  spacing = 50
) {
  let centerCoeff = (layer.count * nodeSize.y + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    addNode(
      cy,
      idPrefix + '_' + layerNum + '_' + nodeNum + 'N',
      'MaxPool2d',
      'MaxPool2d' +
        '\nkernel size: ' +
        layer.kernelSize +
        '\npadding: ' +
        layer.padding +
        '\nstride: ' +
        layer.stride,
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['activation']
    )
  }
  // Возвращает offset для следующего слоя.
  return offset + nodeSize.x + STANDART_GRAPH_GAP
}

function addMergeFlatten(
  cy,
  layer,
  layerNum,
  offset = 0,
  idPrefix = '',
  nodeSize = { x: 100, y: 20 }
) {
  addNode(
    cy,
    idPrefix + '_' + layerNum + '_' + 0 + 'N',
    'MergeFlatten',
    'Merge Flatten',
    nodeSize,
    {
      x: offset + nodeSize.x / 2,
      y: 0
    },
    ['activation']
  )

  // Возвращает offset для следующего слоя.
  return offset + nodeSize.x + STANDART_GRAPH_GAP
}

function addNode(cy, id, type, value, size, pos, classes, constValues = '') {
  cy.add({
    group: 'nodes',
    data: {
      id: id,
      type: type,
      constValues: constValues,
      value: value,
      width: size.x,
      height: size.y
    },
    position: {
      x: pos.x,
      y: pos.y
    },
    locked: true,
    classes: classes
  })
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

function addNNframe(cy, name, lossValue, offsets, idPrefix) {
  // Находим первый добавленый элемент графа, чтобы вычислить отступ влево.
  let elem = cy.getElementById(idPrefix + '_' + 0 + '_' + 0 + 'N')
  let leftTop = [elem.position().x - elem.data('width') / 2, -offsets[1]]
  let center = [offsets[0] + leftTop[0] / 2, 0]

  cy.add({
    group: 'nodes',
    data: {
      id: idPrefix + '_model',
      type: 'Model',
      value: name + '\nloss: ' + lossValue,
      width: offsets[0] / 2,
      height: offsets[1] / 2
    },
    position: {
      x: center[0],
      y: center[1]
    },
    locked: true,
    classes: ['activation']
  })
}
