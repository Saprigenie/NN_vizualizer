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

      <div class="vr ms-2 me-2"></div>

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

      <div class="vr ms-2 me-2"></div>
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
import { graphStyles } from './styleGraph'

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
    style: graphStyles
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
