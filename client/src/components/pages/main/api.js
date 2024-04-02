import { saveAs } from 'file-saver'

import { api, baseURL } from '@/api'
import { drawGraph } from './drawGraph'
import { updateWeights, updateLoss } from './updateWeights'
import { ToastTypes } from '@/store/toaster'

// Запомненный прошлый нажатый узел.
let prevTapNode = null
// Текущие, пришедшие с сервера, веса нейронок для отображения (forward или backward).
let nnStates = {
  ann: null,
  cnn: null,
  generator: null,
  discriminator: null
}

export async function setGraphElements(cy, nnName) {
  // С сервера приходит список с узлами и связями.
  let graphData = (await api.get('/nn/state/' + nnName)).data
  let offset = 0

  for (let i = 0; i < graphData.length; i++) {
    offset = drawGraph(
      cy,
      graphData[i].structure,
      graphData[i].model,
      graphData[i].loss,
      offset,
      graphData[i].model
    )
  }

  addGraphHandlers(cy)
}

export async function nnForwardServer(cy, nnName) {
  // С сервера приходят веса выбранной нейронной сети для отображения (forward или backward).
  if (!nnStates[nnName] || nnStates[nnName].ended) {
    nnStates[nnName] = (await api.get('/nn/train/' + nnName)).data
  }
  if (nnStates[nnName].type == 'forward') {
    // Запоминаем некоторые поля для краткости записи.
    let dataIndex = nnStates[nnName].forwardWeights.dataIndex // Сервер присылает веса сразу для батча обработанных данных.
    let layerIndex = nnStates[nnName].forwardWeights.layerIndex // Индекс слоя, среди тех, которые нужно обновить.
    let weights = nnStates[nnName].forwardWeights.weights[dataIndex][layerIndex].w
    let graphLayerIndex =
      nnStates[nnName].forwardWeights.weights[dataIndex][layerIndex].graphLayerIndex // Индекс слоя, среди всех слоев нейронной сети.
    updateWeights(cy, graphLayerIndex, weights, nnStates[nnName].model)

    // Считаем индекс следующего отображаемого слоя.
    layerIndex += 1
    nnStates[nnName].forwardWeights.layerIndex += 1
    // Переходим на новый набор данных в батче, если уже обновили все слои для одного набора.
    if (layerIndex >= nnStates[nnName].forwardWeights.weights[dataIndex].length) {
      nnStates[nnName].forwardWeights.layerIndex = 0
      dataIndex += 1
      nnStates[nnName].forwardWeights.dataIndex += 1
      let oldData = nnStates[nnName].trainStep.data
      nnStates[nnName].trainStep.data = { curr: oldData.curr + 1, max: oldData.max }
    }
    // Если все данные пройдены, то ставим флаг завершения.
    if (dataIndex >= nnStates[nnName].forwardWeights.weights.length) {
      nnStates[nnName].type = 'backward'
      nnStates[nnName].trainStep.data.curr -= 1
    }
  } else if (nnStates[nnName].type == 'backward') {
    // Запоминаем некоторые поля для краткости записи.
    let layerIndex = nnStates[nnName].backwardWeights.layerIndex
    let weights = nnStates[nnName].backwardWeights.weights[layerIndex].w
    let graphLayerIndex = nnStates[nnName].backwardWeights.weights[layerIndex].graphLayerIndex // Индекс слоя, среди всех слоев нейронной сети.
    updateWeights(cy, graphLayerIndex, weights, nnStates[nnName].model)

    // По всем наборам данных из батча уже прошлись.
    let oldData = nnStates[nnName].trainStep.data
    nnStates[nnName].trainStep.data = { curr: oldData.max, max: oldData.max }

    // Считаем индекс следующего отображаемого слоя.
    layerIndex += 1
    nnStates[nnName].backwardWeights.layerIndex += 1
    // Если все данные пройдены, то ставим флаг завершения.
    if (layerIndex >= nnStates[nnName].backwardWeights.weights.length) {
      nnStates[nnName].ended = true
      updateLoss(cy, nnStates[nnName].loss, nnStates[nnName].model)
    }
  }

  // Возвращаем текущий шаг обучения.
  return nnStates[nnName].trainStep
}

export async function setBatchSizeServer(nnName, batchSize, toaster) {
  await api.put('/nn/batch_size/' + nnName + '/' + batchSize).then((res) => {
    // Добавляем сообщение пользователю.
    toaster.addToast({
      title: 'Информация',
      body: 'Размер батча успешно обновлен.',
      type: ToastTypes.success
    })
  })
}

export async function getBatchSizeServer(nnName) {
  return (await api.get('/nn/batch_size/' + nnName)).data
}

export async function nnRestartServer(cy, nnName, toaster) {
  await api.put('/nn/restart/' + nnName).then((res) => {
    // Добавляем сообщение пользователю.
    toaster.addToast({
      title: 'Информация',
      body: 'Модель ' + nnName + ' полностью обнулена.',
      type: ToastTypes.success
    })
  })

  // Пересоздаем граф.
  cy.elements().remove()
  nnStates[nnName] = null
  setGraphElements(cy, nnName)
}

export async function downloadWeightsServer(nnName) {
  saveAs(baseURL + '/nn/weights/' + nnName, nnName + '.pth')
}

export async function uploadWeightsServer(cy, nnName, file, toaster) {
  let formData = new FormData()
  formData.append('weights', { pupa: 'lupa' })
  formData.append('weights', file)
  await api.post('/nn/weights/' + nnName, formData).then((res) => {
    // Добавляем сообщение пользователю.
    toaster.addToast({
      title: 'Информация',
      body: 'Веса модели ' + nnName + ' успешно обновлены.',
      type: ToastTypes.success
    })
  })

  // Пересоздаем граф.
  cy.elements().remove()
  setGraphElements(cy, nnName)
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
