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
            <input id="setWeightInp" type="file" style="display: none" />
            <button class="dropdown-item nav-link" v-on:click="showWeightsChooser()">
              Сохранить веса модели
            </button>
          </li>

          <li>
            <button class="dropdown-item nav-link" v-on:click="downloadWeightsServer(nnNameChoice)">
              Загрузить веса модели
            </button>
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
          Модель NN: {{ nnNameChoice.toUpperCase() }}
        </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton1">
          <li>
            <button class="dropdown-item nav-link" v-on:click="reloadNN(cy, 'ann')">
              ANN (Полносвязная нейронная сеть)
            </button>
          </li>

          <li>
            <button class="dropdown-item nav-link" v-on:click="reloadNN(cy, 'cnn')">
              CNN (Сверточная нейронная сеть)
            </button>
          </li>

          <li>
            <button class="dropdown-item nav-link" v-on:click="reloadNN(cy, 'gan')">
              GAN (Генеративная нейронная сеть)
            </button>
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
          Размер батча: {{ batchSize }}
        </button>
        <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton2">
          <li>
            <button class="dropdown-item nav-link" v-on:click="setBatchSize(1)">1</button>
          </li>
          <li>
            <button class="dropdown-item nav-link" v-on:click="setBatchSize(2)">2</button>
          </li>
          <li>
            <button class="dropdown-item nav-link" v-on:click="setBatchSize(5)">5</button>
          </li>
          <li>
            <button class="dropdown-item nav-link" v-on:click="setBatchSize(10)">10</button>
          </li>
          <li>
            <button class="dropdown-item nav-link" v-on:click="setBatchSize(64)">64</button>
          </li>
        </ul>
      </li>

      <div class="vr ms-2 me-2"></div>

      <li class="nav-item">
        <button
          class="btn float-start"
          v-on:click="cy.fit()"
          data-bs-toggle="tooltip"
          data-bs-placement="bottom"
          title="Возвращает камеру для отображения структуры нейронной сети."
        >
          <i class="bi bi bi-eye"></i>
        </button>
      </li>
    </ul>
  </nav>

  <div id="cy"></div>

  <div class="card trainStep">
    <div class="row text-center p-2">
      <p>
        {{ trainStep.data.curr }}/{{ trainStep.data.max }} данных, {{ trainStep.batch.curr }}/{{
          trainStep.batch.max
        }}
        батчей, {{ trainStep.epoch.curr }} эпоха
      </p>
    </div>
  </div>

  <footer class="footer bg-dark">
    <div class="row text-center p-2">
      <div class="col">
        <!-- <button
          class="btn btn-dark float-start"
          data-bs-toggle="tooltip"
          data-bs-placement="top"
          title="Возвращает на 1 эпоху обучения нейронной сети назад."
        >
          <i class="bi bi-chevron-double-left text-light"></i>
        </button>
        <button
          class="btn btn-dark float-start"
          data-bs-toggle="tooltip"
          data-bs-placement="top"
          title="Пропускает 1 эпоху обучения нейронной сети."
        >
          <i class="bi bi-chevron-double-right text-light"></i>
        </button> -->
      </div>
      <div class="col">
        <button
          class="btn btn-dark"
          data-bs-toggle="tooltip"
          data-bs-placement="top"
          title="Совершает 1 шаг назад прохода данных по нейронной сети (в пределах 1 батча)."
        >
          <i class="bi bi-arrow-left text-light"></i>
        </button>
        <button
          class="btn btn-dark"
          v-on:click="nnForward()"
          data-bs-toggle="tooltip"
          data-bs-placement="top"
          title="Совершает 1 шаг прохода данных по нейронной сети."
        >
          <i class="bi bi-arrow-right text-light"></i>
        </button>
      </div>
      <div class="col">
        <button
          class="btn btn-dark float-end"
          v-on:click="nnRestartServer(cy, nnNameChoice)"
          data-bs-toggle="tooltip"
          data-bs-placement="top"
          title="Полностью сбрасывает текущий процесс обучения нейронной сети."
        >
          <i class="bi bi-arrow-clockwise"></i>
        </button>
      </div>
    </div>
  </footer>
</template>

<script setup>
import cytoscape from 'cytoscape'
import * as bootstrap from 'bootstrap'
import { onMounted, reactive, ref } from 'vue'
import {
  setGraphElements,
  nnForwardServer,
  setBatchSizeServer,
  getBatchSizeServer,
  nnRestartServer,
  downloadWeightsServer
} from './api'
import { graphStyles } from './styleGraph'

let cy
// Выбранный номер нейронной сети.
let nnNameChoice = ref('ann')
// Выбранный размера батча.
let batchSize = ref(2)
// Текущий шаг обучения.
let trainStep = reactive({
  data: { curr: '?', max: '?' },
  batch: { curr: '?', max: '?' },
  epoch: { curr: '?' }
})

onMounted(() => {
  cy = cytoscape({
    container: document.getElementById('cy'), // container to render in
    style: graphStyles
  })

  reloadNN(cy, nnNameChoice.value)

  // Активировать всплывающие подсказки.
  let tooltipTriggerList = Array.prototype.slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
  )
  tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl)
  })
})

function showWeightsChooser() {
  document.getElementById('setWeightInp').click()
}

async function reloadNN(cy, newChoice) {
  cy.elements().remove()
  nnNameChoice.value = newChoice
  setGraphElements(cy, nnNameChoice.value)
  batchSize.value = await getBatchSizeServer(nnNameChoice.value)
}

async function nnForward() {
  let newTrainStep = await nnForwardServer(cy, nnNameChoice.value)
  trainStep.data = newTrainStep.data
  trainStep.batch = newTrainStep.batch
  trainStep.epoch = newTrainStep.epoch
}

function setBatchSize(newBatchSize) {
  batchSize.value = newBatchSize
  setBatchSizeServer(nnNameChoice.value, batchSize.value)
}
</script>

<style scoped>
#cy {
  width: 100%;
  height: 550px;
}

.trainStep {
  position: fixed;
  bottom: 50px;
  right: 0;
  width: 300px;
  height: 40px;
}

footer {
  position: fixed;
  bottom: 0;
  width: 100%;
  height: 50px;
}
</style>
