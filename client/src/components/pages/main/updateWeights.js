import { formElemId } from './drawGraph'
import { getMax } from '@/api/utility'

// Запомненный прошлый слой, у которого изменялись веса.
let prevHighlightedNodes = null

export function updateWeights(cy, graphLayerIndex, weights, idPrefix) {
  // Находим элементы, у которых необходимо найти веса.
  let changeWeightsElements = cy.filter(function (element, i) {
    let id = element.data('id')
    return id.startsWith(idPrefix + '_' + graphLayerIndex + '_')
  })

  // Убираем подсветку с прошлых подсвеченных.
  if (prevHighlightedNodes) {
    prevHighlightedNodes.removeClass('highlight')
  }
  prevHighlightedNodes = changeWeightsElements
  // Подсвечиваем их.
  changeWeightsElements.addClass('highlight')

  // В зависимости от типа элемента обновляем веса:
  let type = changeWeightsElements[0].data('type')
  switch (type) {
    case 'Data':
      updateData(cy, type, graphLayerIndex, weights, idPrefix)
      break
    case 'DataImage':
      updateDataImage(cy, type, graphLayerIndex, weights, idPrefix)
      break
    case 'Linear':
      updateLinear(cy, type, graphLayerIndex, weights, idPrefix)
      break
    case 'Conv2d':
      updateConv2d(cy, type, graphLayerIndex, weights, idPrefix)
      break
    case 'Connection':
      updateConnection(cy, type, graphLayerIndex, weights, idPrefix)
      break
  }
}

export function updateLoss(cy, loss, idPrefix) {
  let nnFrameElem = cy.getElementById(formElemId('Model', { idPrefix: idPrefix }))
  let values = nnFrameElem.data('values')
  values.loss = loss
  nnFrameElem.data('values', values)
}

function updateData(cy, type, graphLayerIndex, weights, idPrefix) {
  for (let i = 0; i < weights.length; i++) {
    let elem = cy.getElementById(
      formElemId(type, { idPrefix: idPrefix, layerNum: graphLayerIndex, nodeNum: i })
    )
    let values = elem.data('values')
    values.weight = weights[i]
    values.maxWeight = getMax(weights)
    elem.data('values', values)
  }
}

function updateDataImage(cy, type, graphLayerIndex, weights, idPrefix) {
  for (let imageNum = 0; imageNum < weights.length; imageNum++) {
    for (let h = 0; h < weights[imageNum].length; h++) {
      for (let w = 0; w < weights[imageNum][h].length; w++) {
        let elem = cy.getElementById(
          formElemId(type + 'Cell', {
            idPrefix: idPrefix,
            layerNum: graphLayerIndex,
            nodeNum: imageNum,
            w: w,
            h: h
          })
        )
        let values = elem.data('values')
        values.weight = weights[imageNum][h][w]
        values.maxWeight = getMax(weights)
        elem.data('values', values)
      }
    }
  }
}

function updateLinear(cy, type, graphLayerIndex, weights, idPrefix) {
  for (let i = 0; i < weights.length; i++) {
    let elem = cy.getElementById(
      formElemId(type, { idPrefix: idPrefix, layerNum: graphLayerIndex, nodeNum: i })
    )
    let values = elem.data('values')
    values.bias = weights[i]
    elem.data('values', values)
  }
}

function updateConv2d(cy, type, graphLayerIndex, weightsBias, idPrefix) {
  let bias = weightsBias[1]
  let weights = weightsBias[0]
  // Обновляем веса.
  for (let convNum = 0; convNum < weights.length; convNum++) {
    for (let h = 0; h < weights[convNum].length; h++) {
      for (let w = 0; w < weights[convNum][h].length; w++) {
        let elem = cy.getElementById(
          formElemId(type + 'Cell', {
            idPrefix: idPrefix,
            layerNum: graphLayerIndex,
            nodeNum: convNum,
            w: w,
            h: h
          })
        )
        let values = elem.data('values')
        values.weight = weights[convNum][h][w]
        values.maxWeight = getMax(weights)
        elem.data('values', values)
      }
    }
  }

  // Обновляем bias.
  for (let i = 0; i < bias.length; i++) {
    let elem = cy.getElementById(
      formElemId(type, { idPrefix: idPrefix, layerNum: graphLayerIndex, nodeNum: i })
    )
    let values = elem.data('values')
    values.bias = bias[i]
    elem.data('values', values)
  }
}

function updateConnection(cy, type, graphLayerIndex, weights, idPrefix) {
  // Получаем списки узлов, которые были соединены с помощью слоя связей.
  let sources = cy.filter(function (element, i) {
    let id = element.id()
    return element.isNode() && id.startsWith(idPrefix + '_' + (graphLayerIndex - 1) + '_')
  })
  let targets = cy.filter(function (element, i) {
    let id = element.id()
    return element.isNode() && id.startsWith(idPrefix + '_' + (graphLayerIndex + 1) + '_')
  })

  for (let targNum = 0; targNum < targets.length; targNum++) {
    if (Array.isArray(weights[targNum])) {
      // Значит каждый узел прошлого слоя связан с каждым узлом следующего.
      for (let sourceNum = 0; sourceNum < sources.length; sourceNum++) {
        let elem = cy.getElementById(
          formElemId(type, {
            idPrefix: idPrefix,
            connNum: graphLayerIndex,
            sourceNum: sourceNum,
            targetNum: targNum
          })
        )
        let values = elem.data('values')
        values.weight = weights[targNum][sourceNum]
        elem.data('values', values)
      }
    }
  }
}
