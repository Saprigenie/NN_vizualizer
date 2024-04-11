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
  gan: null
}
let nnOldStates = {
  ann: null,
  cnn: null,
  gan: null
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

export async function nnForwardServer(cy, nnNameServer) {
  // Некоторые переменные для удобства доступа.
  let state = nnStates[nnNameServer]
  let modelName = state ? state.model : nnNameServer

  // Сохраняем текущее состояние в старое, когда оно завершено.
  if (state && state.ended) {
    nnOldStates[modelName] = state
  }
  // С сервера приходят веса выбранной нейронной сети для отображения (forward или backward).
  if (!state || state.ended) {
    nnStates[nnNameServer] = (await api.get('/nn/train/' + nnNameServer)).data
    state = nnStates[nnNameServer]
    modelName = state.model
  }

  // Некоторые переменные для удобства доступа.
  let forwardWeights = state.forwardWeights
  let backwardWeights = state.backwardWeights
  let oldData = state.trainStep.data

  if (state.type == 'forward') {
    let layer = forwardWeights.weights[forwardWeights.dataIndex][forwardWeights.layerIndex]
    updateWeights(cy, layer.graphLayerIndex, layer.w, modelName)

    // Считаем индекс следующего отображаемого слоя.
    forwardWeights.layerIndex += 1
    // Переходим на новый набор данных в батче, если уже обновили все слои для одного набора.
    if (forwardWeights.layerIndex >= forwardWeights.weights[forwardWeights.dataIndex].length) {
      forwardWeights.layerIndex = 0
      forwardWeights.dataIndex += 1
      state.trainStep.data = { curr: oldData.curr + 1, max: oldData.max }
    }
    // Если все данные пройдены, то ставим флаг завершения.
    if (forwardWeights.dataIndex >= forwardWeights.weights.length) {
      state.type = 'backward'
      state.trainStep.data.curr -= 1
    }
  } else if (state.type == 'backward') {
    let layer = backwardWeights.weights[backwardWeights.layerIndex]
    updateWeights(cy, layer.graphLayerIndex, layer.w, modelName)

    // По всем наборам данных из батча уже прошлись.
    state.trainStep.data = { curr: oldData.max, max: oldData.max }

    // Считаем индекс следующего отображаемого слоя.
    backwardWeights.layerIndex += 1
    // Если все данные пройдены, то ставим флаг завершения.
    if (backwardWeights.layerIndex >= backwardWeights.weights.length) {
      state.ended = true
      updateLoss(cy, state.loss, modelName)
    }
  }
  // Считаем условие для активации клавиши с возможностью обратно откатить шаги.
  let newBackEnable =
    nnOldStates[modelName] &&
    (state.type == 'backward' || forwardWeights.dataIndex > 0 || forwardWeights.layerIndex > 1)

  // Возвращаем текущий шаг обучения.
  return { newBackEnable: newBackEnable, newTrainStep: state.trainStep }
}

export function nnBackServer(cy, nnNameServer) {
  // Некоторые переменные для удобства доступа.
  let state = nnStates[nnNameServer]
  let modelName = state.model
  let oldState = nnOldStates[modelName]
  let forwardWeights = state.forwardWeights
  let backwardWeights = state.backwardWeights
  let oldData = state.trainStep.data

  // Откатываемся назад, поэтому шаги по текущему батчу еще точно не завершены
  state.ended = false

  // Внутренняя функция, которая на 1 шаг назад откатывает по прямому проходу по nn.
  function forwardBack() {
    // Считаем индекс прошлого отображаемого слоя.
    forwardWeights.layerIndex -= 1

    // Переходим на старый набор данных в батче, если уже обновили все слои для одного набора.
    if (forwardWeights.layerIndex < 0) {
      forwardWeights.dataIndex -= 1
      forwardWeights.layerIndex = forwardWeights.weights[forwardWeights.dataIndex].length - 1
      state.trainStep.data = { curr: oldData.curr - 1, max: oldData.max }
    }

    // Обновляем последний слой старыми весами.
    let oldLayer = forwardWeights.dataIndex
      ? forwardWeights.weights[forwardWeights.dataIndex - 1][forwardWeights.layerIndex]
      : oldState.forwardWeights.weights[oldState.forwardWeights.weights.length - 1][
          forwardWeights.layerIndex
        ]
    updateWeights(cy, oldLayer.graphLayerIndex, oldLayer.w, modelName)
    // Подсвечиваем последний ОБНОВЛЕННЫЙ слой.
    let oldOldLayer = forwardWeights.layerIndex
      ? forwardWeights.weights[forwardWeights.dataIndex][forwardWeights.layerIndex - 1]
      : forwardWeights.weights[forwardWeights.dataIndex - 1][
          forwardWeights.weights[forwardWeights.dataIndex - 1].length - 1
        ]
    updateWeights(cy, oldOldLayer.graphLayerIndex, oldOldLayer.w, modelName)
  }

  // Внутренняя функция, которая на 1 шаг назад откатывает по обратному проходу по nn.
  function backwardBack() {
    // Считаем индекс прошлого отображаемого слоя.
    backwardWeights.layerIndex -= 1

    // Если вернулись после того, как все данные пройдены.
    if (backwardWeights.layerIndex == backwardWeights.weights.length - 1) {
      updateLoss(cy, oldState.loss, oldState.model)
    }

    // Обновляем последний слой старыми весами.
    let oldLayer = oldState.backwardWeights.weights[backwardWeights.layerIndex]
    updateWeights(cy, oldLayer.graphLayerIndex, oldLayer.w, modelName)
    // Подсвечиваем последний ОБНОВЛЕННЫЙ слой.
    let oldOldLayer = backwardWeights.layerIndex
      ? oldState.backwardWeights.weights[backwardWeights.layerIndex - 1]
      : forwardWeights.weights[forwardWeights.weights.length - 1][
          forwardWeights.weights[forwardWeights.weights.length - 1].length - 1
        ]
    updateWeights(cy, oldOldLayer.graphLayerIndex, oldOldLayer.w, modelName)

    // По всем наборам данных из батча уже прошлись.
    state.trainStep.data = { curr: oldData.max, max: oldData.max }
  }

  if (state.type == 'forward') {
    forwardBack()
  } else if (state.type == 'backward') {
    // Если мы собираемся обратно откатываться по обратному проходу, но при этом мы только на него переключились, то возвращаемся
    // на форвард.
    if (backwardWeights.layerIndex == 0) {
      state.type = 'forward'
      forwardBack()
      state.trainStep.data = { curr: oldData.max, max: oldData.max }
    } else {
      backwardBack()
    }
  }
  // Считаем условие для активации клавиши с возможностью обратно откатить шаги.
  let newBackEnable =
    state.type == 'backward' || forwardWeights.dataIndex > 0 || forwardWeights.layerIndex > 1

  // Возвращаем текущий шаг обучения.
  return { newBackEnable: newBackEnable, newTrainStep: state.trainStep }
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
  if (nnStates[nnName]) nnOldStates[nnStates[nnName].model] = null
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
