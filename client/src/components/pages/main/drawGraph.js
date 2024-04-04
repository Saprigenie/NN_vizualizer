import { STANDART_GRAPH_GAP } from '@/constants'
import { getMax } from '@/api/utility'

export function drawGraph(cy, graphData, name, loss, offset = 0, idPrefix = '') {
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
      case 'Reshape':
        offset = addReshape(cy, layer, layerNum, offset, idPrefix)
        break
    }
  }
  // Добавляем связи.
  for (let layerNum = 0; layerNum < graphData.length; layerNum++) {
    let layer = graphData[layerNum]
    if (layer.type === 'Connection') {
      addConnection(cy, layer, layerNum, idPrefix)
    }
  }

  // Добавляем рамку с названием нейронной сети и loss-ом.
  addNNframe(cy, name, loss, idPrefix)

  // Возвращаем занятое нейронной сетью пространство.
  return offset + STANDART_GRAPH_GAP
}

export function formElemId(type, params) {
  switch (type) {
    case 'Data':
    case 'DataImage':
    case 'Linear':
    case 'Activation':
    case 'Conv2d':
    case 'MaxPool2d':
    case 'MergeFlatten':
    case 'Reshape':
      return params.idPrefix + '_' + params.layerNum + '_' + params.nodeNum + 'N'
    case 'DataImageCell':
    case 'Conv2dCell':
      return (
        params.idPrefix +
        '_cell_' +
        params.layerNum +
        '_' +
        params.nodeNum +
        '_' +
        params.h +
        '_' +
        params.w +
        'N'
      )
    case 'Connection':
      return (
        params.idPrefix +
        '_' +
        params.connNum +
        '_' +
        params.sourceNum +
        '_' +
        params.targetNum +
        'E'
      )
    case 'Model':
      return params.idPrefix + '_model'
  }
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
  // Для отображения более ярким цветом элекментов с большим значением веса в слое.
  let maxWeight = getMax(layer.weights)

  let centerCoeff = (layer.count * nodeSize.y + spacing * (layer.count - 1)) / 2
  for (let nodeNum = 0; nodeNum < layer.count; nodeNum++) {
    addNode(
      cy,
      formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: nodeNum }),
      layer.type,
      {
        weight: layer.weights[nodeNum],
        maxWeight: maxWeight
      },
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['data', 'textCenter', 'textContrast', 'border']
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
  // Для отображения более ярким цветом элекментов с большим значением веса в слое.
  let maxWeight = getMax(layer.weights)

  let blocksN = layer.count[0]
  let cols = layer.count[2]
  let rows = layer.count[1]

  let blockSize = { x: rows * nodeSize.x, y: cols * nodeSize.y }

  let centerCoeff = (blocksN * blockSize.y + spacing * (blocksN - 1)) / 2
  for (let currBlock = 0; currBlock < blocksN; currBlock++) {
    // Добавляем 1 node - который в себе будет содержать данные изображения.
    addNode(
      cy,
      formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: currBlock }),
      layer.type,
      {},
      blockSize,
      {
        x: offset + blockSize.x / 2,
        y:
          currBlock * (blockSize.y + spacing) + // offset по количеству пикселей в столбце.
          blockSize.y / 2 -
          centerCoeff // Центруем относительно начального угла обзора.
      },
      ['dataImage', 'textTop', 'textContrast', 'border']
    )

    for (let h = 0; h < rows; h++) {
      for (let w = 0; w < cols; w++) {
        // Добавляем сами данные.
        addNode(
          cy,
          formElemId(layer.type + 'Cell', {
            idPrefix: idPrefix,
            layerNum: layerNum,
            nodeNum: currBlock,
            w: w,
            h: h
          }),
          layer.type + 'Cell',
          {
            weight: layer.weights[currBlock][h][w],
            maxWeight: maxWeight
          },
          nodeSize,
          {
            x: offset + w * nodeSize.x + nodeSize.x / 2,
            y:
              currBlock * (blockSize.y + spacing) + // offset разных изображений в слое.
              h * nodeSize.y + // offset по номеру пикселя в строке.
              nodeSize.y / 2 -
              centerCoeff // Центруем относительно начального угла обзора.
          },
          ['data', 'textCenter', 'textContrast', 'border']
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
      formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: nodeNum }),
      layer.type,
      {
        bias: layer.bias[nodeNum]
      },
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['linear', 'textCenter', 'textWhite', 'blackBorder']
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
      formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: nodeNum }),
      layer.type,
      {
        actType: layer.activation
      },
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['activation', 'textCenter', 'textWhite', 'blackBorder']
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
  // Для отображения более ярким цветом элекментов с большим значением веса в слое.
  let maxWeight = getMax(layer.weights)

  let blocksN = layer.count[0]
  let cols = layer.count[2]
  let rows = layer.count[1]

  let blockSize = { x: rows * nodeSize.x, y: cols * nodeSize.y }

  let centerCoeff = (blocksN * cols * nodeSize.y + spacing * (blocksN - 1)) / 2
  let elementSize = { x: rows * nodeSize.x, y: blockSize.y }
  for (let currBlock = 0; currBlock < blocksN; currBlock++) {
    // Добавляем 1 node - который в себе будет содержать данные всего Conv2d.
    addNode(
      cy,
      formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: currBlock }),
      layer.type,
      {
        padding: layer.padding,
        stride: layer.stride,
        bias: layer.bias[currBlock]
      },
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
      ['convolution', 'textTop', 'textContrast', 'border']
    )

    for (let h = 0; h < rows; h++) {
      for (let w = 0; w < cols; w++) {
        // Добавляем данные фильтра.
        addNode(
          cy,
          formElemId(layer.type + 'Cell', {
            idPrefix: idPrefix,
            layerNum: layerNum,
            nodeNum: currBlock,
            w: w,
            h: h
          }),
          layer.type + 'Cell',
          {
            weight: layer.weights[currBlock][h][w],
            maxWeight: maxWeight
          },
          nodeSize,
          {
            x: offset + w * nodeSize.x + nodeSize.x / 2,
            y:
              currBlock * (blockSize.y + spacing) + // offset разных изображений в слое.
              h * nodeSize.y + // offset по номеру пикселя в строке.
              nodeSize.y / 2 -
              centerCoeff // Центруем относительно начального угла обзора.
          },
          ['data', 'textCenter', 'textContrast', 'border']
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
      formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: nodeNum }),
      layer.type,
      {
        kernelSize: layer.kernelSize,
        padding: layer.padding,
        stride: layer.stride
      },
      nodeSize,
      {
        x: offset + nodeSize.x / 2,
        y: (nodeSize.y + spacing) * nodeNum + nodeSize.y / 2 - centerCoeff // Центруем относительно начального угла обзора.
      },
      ['maxPool2d', 'textCenter', 'textWhite', 'blackBorder']
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
    formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: 0 }),
    layer.type,
    {},
    nodeSize,
    {
      x: offset + nodeSize.x / 2,
      y: 0
    },
    ['mergeFlatten', 'textCenter', 'textWhite', 'blackBorder']
  )

  // Возвращает offset для следующего слоя.
  return offset + nodeSize.x + STANDART_GRAPH_GAP
}

function addReshape(cy, layer, layerNum, offset = 0, idPrefix = '', nodeSize = { x: 100, y: 20 }) {
  addNode(
    cy,
    formElemId(layer.type, { idPrefix: idPrefix, layerNum: layerNum, nodeNum: 0 }),
    layer.type,
    {},
    nodeSize,
    {
      x: offset + nodeSize.x / 2,
      y: 0
    },
    ['reshape', 'textCenter', 'textWhite', 'blackBorder']
  )

  // Возвращает offset для следующего слоя.
  return offset + nodeSize.x + STANDART_GRAPH_GAP
}

function addNode(cy, id, type, values, size, pos, classes) {
  cy.add({
    group: 'nodes',
    data: {
      id: id,
      type: type,
      values: values,
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
          formElemId(connection.type, {
            idPrefix: idPrefix,
            connNum: connectionNum,
            sourceNum: sourceNum,
            targetNum: targNum
          }),
          sources[sourceNum].id(),
          targets[targNum].id(),
          { weight: connection.weights[targNum][sourceNum] }
        )
      }
    } else {
      // Значит каждый узел предыдущего слоя связан с 1 узлом следующего слоя.
      addEdge(
        cy,
        formElemId(connection.type, {
          idPrefix: idPrefix,
          connNum: connectionNum,
          sourceNum: targNum,
          targetNum: targNum
        }),
        sources[targNum].id(),
        targets[targNum].id()
      )
    }
  }
}

function addEdge(cy, id, source, target, values = {}) {
  let edgeParameters = {
    group: 'edges',
    data: {
      type: 'Connection',
      id: id,
      values: values,
      source: source,
      target: target
    },
    locked: true,
    classes: []
  }

  if (Object.keys(edgeParameters.data.values).length !== 0) {
    edgeParameters.classes.push('ehasweights')
  } else {
    edgeParameters.classes.push('enothasweights')
  }

  // Добавляем сформированную связь в граф.
  cy.add(edgeParameters)
}

function addNNframe(cy, name, loss, idPrefix) {
  // Находим минимальные и максимальные значения x и y для отрисовки рамки нейронной сети.
  let allNodes = cy.filter(function (element, i) {
    let id = element.data('id')
    return element.isNode() && id.startsWith(idPrefix + '_')
  })
  let minPoint = { x: allNodes[0].position().x, y: allNodes[0].position().y }
  let maxPoint = { x: allNodes[0].position().x, y: allNodes[0].position().y }

  for (let node of allNodes) {
    let nodeHalfW = node.data('width') / 2
    let nodeHalfH = node.data('height') / 2
    minPoint.x = Math.min(minPoint.x, node.position().x - nodeHalfW)
    minPoint.y = Math.min(minPoint.y, node.position().y - nodeHalfH)
    maxPoint.x = Math.max(maxPoint.x, node.position().x + nodeHalfW)
    maxPoint.y = Math.max(maxPoint.y, node.position().y + nodeHalfH)
  }

  addNode(
    cy,
    formElemId('Model', { idPrefix: idPrefix }),
    'Model',
    { name: name, loss: loss },
    {
      x: maxPoint.x - minPoint.x + STANDART_GRAPH_GAP,
      y: maxPoint.y - minPoint.y + STANDART_GRAPH_GAP
    },
    {
      x: (maxPoint.x + minPoint.x) / 2,
      y: (maxPoint.y + minPoint.y) / 2
    },
    ['model', 'textTop', 'textContrast', 'border']
  )
}
