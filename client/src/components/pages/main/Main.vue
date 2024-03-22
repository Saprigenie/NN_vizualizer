<template>
  <nav class="navbar navbar-expand">
    <ul class="navbar-nav mr-auto">
      <li class="nav-item dropdown">
        <button
          class="nav-link dropdown-toggle ms-2"
          type="button"
          id="dropdownMenuButton1"
          data-bs-toggle="dropdown"
          aria-expanded="false"
        >
          Веса
        </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
          <li>
            <a class="dropdown-item nav-link"> Сохранить веса модели </a>
          </li>

          <li>
            <a class="dropdown-item nav-link"> Загрузить веса модели </a>
          </li>
        </ul>
      </li>

      <li class="nav-item dropdown">
        <button
          class="nav-link dropdown-toggle"
          type="button"
          id="dropdownMenuButton1"
          data-bs-toggle="dropdown"
          aria-expanded="false"
        >
          Выбор модели NN
        </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
          <li>
            <a class="dropdown-item nav-link" v-on:click="reloadNN(cy, 'ann')">
              ANN (Полносвязная нейронная сеть)
            </a>
          </li>

          <li>
            <a class="dropdown-item nav-link" v-on:click="reloadNN(cy, 'cnn')">
              CNN (Сверточная нейронная сеть)
            </a>
          </li>

          <li>
            <a class="dropdown-item nav-link" v-on:click="reloadNN(cy, 'gan')">
              GAN (Генеративная нейронная сеть)
            </a>
          </li>
        </ul>
      </li>

      <div class="vr ms-2 me-2"></div>

      <li class="nav-item dropdown">
        <button
          class="nav-link dropdown-toggle"
          type="button"
          id="dropdownMenuButton2"
          data-bs-toggle="dropdown"
          aria-expanded="false"
        >
          Выбор batch size
        </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton2">
          <li>
            <a class="dropdown-item nav-link" v-on:click="changeBatchSize(nnNameChoice, 1)"> 1 </a>
          </li>
          <li>
            <a class="dropdown-item nav-link" v-on:click="changeBatchSize(nnNameChoice, 2)"> 2 </a>
          </li>
          <li>
            <a class="dropdown-item nav-link" v-on:click="changeBatchSize(nnNameChoice, 5)"> 5 </a>
          </li>
          <li>
            <a class="dropdown-item nav-link" v-on:click="changeBatchSize(nnNameChoice, 10)">
              10
            </a>
          </li>
          <li>
            <a class="dropdown-item nav-link" v-on:click="changeBatchSize(nnNameChoice, 64)">
              64
            </a>
          </li>
        </ul>
      </li>
    </ul>
  </nav>

  <div id="cy"></div>

  <footer class="footer bg-dark">
    <div class="row text-center p-2">
      <div class="col-sm">
        <button class="btn btn-dark float-start">
          <i class="bi bi-chevron-double-left text-light"></i>
        </button>
        <button class="btn btn-dark float-start">
          <i class="bi bi-chevron-double-right text-light"></i>
        </button>
      </div>
      <div class="col-sm mx-auto">
        <button class="btn btn-dark">
          <i class="bi bi-arrow-left text-light"></i>
        </button>
        <button class="btn btn-dark" v-on:click="nnForward(cy, nnNameChoice)">
          <i class="bi bi-arrow-right text-light"></i>
        </button>
      </div>
      <div class="col-sm">
        <button class="btn btn-dark float-end" v-on:click="nnRestart(cy, nnNameChoice)">
          <i class="bi bi-arrow-clockwise"></i>
        </button>
      </div>
    </div>
  </footer>
</template>

<script setup>
import cytoscape from 'cytoscape'
import { onMounted } from 'vue'
import { setGraphElements, nnForward, changeBatchSize, nnRestart } from './api'

let cy
// Выбранный номер нейронной сети.
let nnNameChoice = 'ann'

function reloadNN(cy, newChoice) {
  cy.elements().remove()
  nnNameChoice = newChoice
  setGraphElements(cy, nnNameChoice)
}

onMounted(() => {
  cy = cytoscape({
    container: document.getElementById('cy'), // container to render in
    style: [
      // the stylesheet for the graph
      {
        selector: 'node',
        style: {
          'background-color': '#666',
          content: 'data(value)',
          width: function (elem) {
            return elem.data('width') + 'px'
          },
          height: function (elem) {
            return elem.data('height') + 'px'
          },
          'text-valign': 'center',
          'text-halign': 'center',
          'text-wrap': 'wrap'
        }
      },

      {
        selector: '.data',
        style: {
          shape: 'rectangle',
          'background-color': function (elem) {
            // Меняем цвет данных в зависмости от данных.
            if (elem.data('value') > 0) {
              let value = Math.min(255, parseInt((255 * elem.data('value')) / 16))
              return 'rgb(' + value + ',' + value + ',' + value + ')'
            } else {
              return 'black'
            }
          },
          'text-outline-width': '1',
          'text-outline-color': 'white',
          'border-color': '#666',
          'border-width': '2'
        }
      },

      {
        selector: '.dataImage',
        style: {
          shape: 'rectangle',
          'text-valign': 'top',
          'border-color': '#666',
          'border-width': '3'
        }
      },

      {
        selector: '.linear',
        style: {
          shape: 'round-rectangle',
          color: 'white',
          'border-width': '2'
        }
      },

      {
        selector: '.activation',
        style: {
          shape: 'cut-rectangle',
          color: 'white',
          'border-width': '2'
        }
      },

      {
        selector: '.convolution',
        style: {
          'text-valign': 'top',
          shape: 'cut-rectangle',
          'border-width': '2'
        }
      },

      {
        selector: 'edge',
        style: {
          width: 3,
          'text-outline-width': '1',
          'text-outline-color': 'white',
          'text-outline-opacity': '1'
        }
      },
      {
        selector: '.ehasweights',
        style: {
          'line-color': 'black',
          opacity: '0.1'
        }
      },
      {
        selector: '.enothasweights',
        style: {
          'line-color': '#ccc'
        }
      },
      {
        selector: '.highlight',
        style: {
          'line-color': 'green',
          'border-color': 'green',
          'border-width': '10px'
        }
      },
      {
        selector: '.edisplayweights',
        style: {
          content: 'data(value)',
          'line-color': 'red',
          'line-opacity': 0.5,
          'z-index': 1
        }
      }
    ]
  })

  reloadNN(cy, nnNameChoice)
})
</script>

<style scoped>
#cy {
  width: 100%;
  height: 550px;
}

footer {
  position: fixed;
  bottom: 0;
  width: 100%;
}
</style>
