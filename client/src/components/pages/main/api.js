import { api } from '@/api'
import { drawGraph } from './drawGraph'
import { updateWeights } from './updateWeights'

// Запомненный прошлый нажатый узел.
let prevTapNode = null
// Текущие, пришедшие с сервера, веса нейронок для отображения (forward или backward).
let nnWeights = {
  ann: null,
  cnn: null,
  gan: null
}

export async function setGraphElements(cy, nnName) {
  // С сервера приходит список с узлами и связями.
  let graphData = (await api.get('/nn/state/' + nnName)).data
  let offset = 0

  for (let i = 0; i < graphData.length; i++) {
    offset = drawGraph(cy, graphData[i].structure, offset, graphData[i].model)
  }

  addGraphHandlers(cy)
}

export async function nnForward(cy, nnName) {
  // С сервера приходят веса выбранной нейронной сети для отображения (forward или backward).
  if (!nnWeights[nnName] || nnWeights[nnName].ended) {
    nnWeights[nnName] = (await api.get('/nn/train/' + nnName)).data
    console.log(nnWeights[nnName])
  }
  if (nnWeights[nnName].type == 'forward') {
    // Запоминаем некоторые поля для краткости записи.
    let dataIndex = nnWeights[nnName].dataIndex // Сервер присылает веса сразу для батча обработанных данных.
    let layerIndex = nnWeights[nnName].layerIndex // Индекс слоя, среди тех, которые нужно обновить.
    let weights = nnWeights[nnName].weights[dataIndex][layerIndex].w
    let graphLayerIndex = nnWeights[nnName].weights[dataIndex][layerIndex].graphLayerIndex // Индекс слоя, среди всех слоев нейронной сети.
    updateWeights(cy, graphLayerIndex, weights, nnWeights[nnName].model)

    // Считаем индекс следующего отображаемого слоя.
    layerIndex += 1
    nnWeights[nnName].layerIndex += 1
    if (layerIndex >= nnWeights[nnName].weights[dataIndex].length) {
      nnWeights[nnName].layerIndex = 0
      dataIndex += 1
      nnWeights[nnName].dataIndex += 1
    }
    // Если все данные пройдены, то ставим флаг завершения.
    if (dataIndex >= nnWeights[nnName].weights.length) {
      nnWeights[nnName].ended = true
    }
  } else if (nnWeights[nnName].type == 'backward') {
    // Запоминаем некоторые поля для краткости записи.
    let layerIndex = nnWeights[nnName].layerIndex
    let weights = nnWeights[nnName].weights[layerIndex].w
    let graphLayerIndex = nnWeights[nnName].weights[layerIndex].graphLayerIndex // Индекс слоя, среди всех слоев нейронной сети.
    updateWeights(cy, graphLayerIndex, weights, nnWeights[nnName].model)

    // Считаем индекс следующего отображаемого слоя.
    layerIndex += 1
    nnWeights[nnName].layerIndex += 1
    // Если все данные пройдены, то ставим флаг завершения.
    if (layerIndex >= nnWeights[nnName].weights.length) {
      nnWeights[nnName].ended = true
    }
  }
}

function addGraphHandlers(cy) {
  // Добавляем отображение весов связей при нажатии на линейный слой.
  cy.on('tap', 'node[type = "Linear"]', function (evt) {
    // Если до этого был нажат узел, то нужно снять выделение с его связей.
    if (prevTapNode) {
      let prevSourceEdges = cy.filter(function (element, i) {
        return element.isEdge() && element.data('target') == prevTapNode.id()
      })
      prevSourceEdges.removeClass('edisplayweights')
      prevSourceEdges.addClass('ehasweights')
    }

    let node = evt.target
    let sourceEdges = cy.filter(function (element, i) {
      return element.isEdge() && element.data('target') == node.id()
    })
    sourceEdges.addClass('edisplayweights')
    sourceEdges.removeClass('ehasweights')

    // Запоминаем его.
    prevTapNode = node
  })
}
