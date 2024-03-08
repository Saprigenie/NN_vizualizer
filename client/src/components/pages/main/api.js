import { api } from '@/api'
import { drawGraph } from './nnGraphElems'
import { NN_NAMES } from '@/constants'

// Запомненный прошлый нажатый узел
let prevTapNodeId = ''

export async function setGraphElements(cy, nnInd) {
  // С сервера приходит список с узлами и связями.
  let graphData = (await api.get('/nn/state/' + NN_NAMES[nnInd])).data

  if (nnInd == 2) {
    drawGraph(cy, graphData.generator, 0, '1_')
    drawGraph(cy, graphData.discriminator, 6000, '2_')
  } else {
    drawGraph(cy, graphData, 0)
  }

  addGraphHandlers(cy)
}

function addGraphHandlers(cy) {
  // Добавляем отображение весов связей при нажатии на линейный слой.
  cy.on('tap', 'node[type = "linear"]', function (evt) {
    // Если до этого был нажат узел, то нужно снять выделение с его связей.
    if (prevTapNodeId !== '') {
      let prevSourceEdges = cy.filter(function (element, i) {
        return element.isEdge() && element.data('target') == prevTapNodeId
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

    // Запоминаем его id.
    prevTapNodeId = node.id()
  })
}
