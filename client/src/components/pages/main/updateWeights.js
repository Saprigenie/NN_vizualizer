// Запомненный прошлый слой, у которого изменялись веса.
let prevHighlightedNodes = null

export function updateWeights(cy, graphLayerIndex, weights, idPrefix) {
  // Находим элементы, у которых необходимо найти веса.
  let changeWeightsElements = cy.filter(function (element, i) {
    let id = element.data('id')
    return id.startsWith(idPrefix + '_' + graphLayerIndex + '_')
  })
  // Подсвечиваем их.
  changeWeightsElements.addClass('highlight')
  if (prevHighlightedNodes) {
    prevHighlightedNodes.removeClass('highlight')
  }
  prevHighlightedNodes = changeWeightsElements

  // В зависимости от типа элемента обновляем веса:
  let type = changeWeightsElements[0].data('type')
  switch (type) {
    case 'Data':
      updateData(cy, graphLayerIndex, weights, idPrefix)
      break
    case 'DataImage':
      updateDataImage(cy, graphLayerIndex, weights, idPrefix)
      break
    case 'Linear':
      updateLinear(cy, graphLayerIndex, weights, idPrefix)
      break
    case 'Conv2d':
      updateConv2d(cy, graphLayerIndex, weights, idPrefix)
      break
    case 'Connection':
      updateConnection(cy, graphLayerIndex, weights, idPrefix)
      break
  }
}

function updateData(cy, graphLayerIndex, weights, idPrefix) {
  for (let i = 0; i < weights.length; i++) {
    let elem = cy.getElementById(idPrefix + '_' + graphLayerIndex + '_' + i + 'N')
    elem.data('value', Number.parseFloat(weights[i]).toFixed(3))
  }
}

function updateDataImage(cy, graphLayerIndex, weights, idPrefix) {
  for (let imageNum = 0; imageNum < weights.length; imageNum++) {
    for (let h = 0; h < weights[imageNum].length; h++) {
      for (let w = 0; w < weights[imageNum][h].length; w++) {
        let elem = cy.getElementById(
          idPrefix + '_image_' + imageNum + '_' + graphLayerIndex + '_' + h + '_' + w + 'N'
        )
        elem.data('value', Number.parseFloat(weights[imageNum][h][w]).toFixed(3))
      }
    }
  }
}

function updateLinear(cy, graphLayerIndex, weights, idPrefix) {
  for (let i = 0; i < weights.length; i++) {
    let elem = cy.getElementById(idPrefix + '_' + graphLayerIndex + '_' + i + 'N')
    elem.data('value', 'Linear\nbias: ' + Number.parseFloat(weights[i]).toFixed(4))
  }
}

function updateConv2d(cy, graphLayerIndex, weightsBias, idPrefix) {
  let bias = weightsBias[1]
  let weights = weightsBias[0]
  // Обновляем веса.
  for (let convNum = 0; convNum < weights.length; convNum++) {
    for (let h = 0; h < weights[convNum].length; h++) {
      for (let w = 0; w < weights[convNum][h].length; w++) {
        let elem = cy.getElementById(
          idPrefix + '_image_' + convNum + '_' + graphLayerIndex + '_' + h + '_' + w + 'N'
        )
        elem.data('value', Number.parseFloat(weights[convNum][h][w]).toFixed(3))
      }
    }
  }

  // Обновляем bias.
  for (let i = 0; i < bias.length; i++) {
    let elem = cy.getElementById(idPrefix + '_' + graphLayerIndex + '_' + i + 'N')
    elem.data(
      'value',
      elem.data('constValues') + '\nbias: ' + Number.parseFloat(bias[i]).toFixed(4)
    )
  }
}

function updateConnection(cy, graphLayerIndex, weights, idPrefix) {
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
          idPrefix + '_' + graphLayerIndex + '_' + sourceNum + '_' + targNum + 'E'
        )
        elem.data('value', Number.parseFloat(weights[targNum][sourceNum]).toFixed(4))
      }
    }
  }
}
