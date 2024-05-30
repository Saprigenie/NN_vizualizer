<template>
  <nav class="navbar navbar-expand-lg">
    <div class="container-fluid">
      <button
        class="navbar-toggler"
        type="button"
        data-bs-toggle="collapse"
        data-bs-target="#navbarSupportedContent"
        aria-controls="navbarSupportedContent"
        aria-expanded="false"
        aria-label="Toggle navigation"
      >
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarSupportedContent">
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
                <button
                  class="dropdown-item nav-link"
                  v-on:click="downloadWeightsServer(nnNameChoice)"
                >
                  Сохранить текущие веса модели
                </button>
              </li>

              <li>
                <input
                  id="setWeightInp"
                  type="file"
                  style="display: none"
                  v-on:change="
                    (ev) => uploadWeightsServer(cy, nnNameChoice, ev.target.files[0], toaster)
                  "
                />
                <button class="dropdown-item nav-link" v-on:click="showWeightsChooser()">
                  Загрузить новые веса модели
                </button>
              </li>
            </ul>
          </li>

          <div class="vr d-none d-lg-block ms-2 me-2"></div>

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
                <button class="dropdown-item nav-link" v-on:click="reloadNN(cy, 'smallann')">
                  Small ANN (Маленькая версия полносвязной нейронной сети)
                </button>
              </li>
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

          <li class="nav-item dropdown">
            <button
              class="nav-link dropdown-toggle"
              type="button"
              id="dropdownMenuButton2"
              data-bs-toggle="dropdown"
              aria-expanded="false"
            >
              Скорость обучения: {{ learnRate }}
            </button>
            <ul class="dropdown-menu" aria-labelledby="dropdownMenuButton2">
              <li>
                <button class="dropdown-item nav-link" v-on:click="setLearnRate(0.025)">
                  0.025
                </button>
              </li>
              <li>
                <button class="dropdown-item nav-link" v-on:click="setLearnRate(0.05)">0.05</button>
              </li>
              <li>
                <button class="dropdown-item nav-link" v-on:click="setLearnRate(0.1)">0.1</button>
              </li>
              <li>
                <button class="dropdown-item nav-link" v-on:click="setLearnRate(0.5)">0.5</button>
              </li>
              <li>
                <button class="dropdown-item nav-link" v-on:click="setLearnRate(1)">1</button>
              </li>
            </ul>
          </li>

          <div class="vr d-none d-lg-block ms-2 me-2"></div>

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
      </div>
    </div>
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

  <footer class="footer container-fluid bg-dark">
    <div class="row text-center">
      <div class="col"></div>
      <div class="col">
        <button
          class="btn btn-dark"
          v-on:click="nnBack()"
          :disabled="!backEnable"
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
          v-on:click="nnRestart()"
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
  nnBackServer,
  setBatchSizeServer,
  setLearnRateServer,
  getBatchSizeServer,
  getLearnRateServer,
  nnRestartServer,
  downloadWeightsServer,
  uploadWeightsServer
} from './api'
import { useToaster } from '@/store/toaster'
import { graphStyles } from './styleGraph'

// Будующий экземпляр холста, на котором строится график cytospace.
let cy
// Выбранный номер нейронной сети.
let nnNameChoice = ref('ann')
// Выбранный размера батча.
let batchSize = ref(2)
// Выбранный размер скорости обучения.
let learnRate = ref(0.05)
// Текущий шаг обучения.
let trainStep = reactive({
  data: { curr: '?', max: '?' },
  batch: { curr: '?', max: '?' },
  epoch: { curr: '?' }
})
// Включена или нет клавиша назад на шаг.
let backEnable = ref(false)
// Для отображения сообщений пользователю.
const toaster = useToaster()

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
  learnRate.value = await getLearnRateServer(nnNameChoice.value)
  backEnable.value = false
}

async function nnForward() {
  let { newBackEnable, newTrainStep } = await nnForwardServer(cy, nnNameChoice.value)
  trainStep.data = newTrainStep.data
  trainStep.batch = newTrainStep.batch
  trainStep.epoch = newTrainStep.epoch
  backEnable.value = newBackEnable
}

async function nnBack() {
  let { newBackEnable, newTrainStep } = nnBackServer(cy, nnNameChoice.value)
  trainStep.data = newTrainStep.data
  trainStep.batch = newTrainStep.batch
  trainStep.epoch = newTrainStep.epoch
  backEnable.value = newBackEnable
}

function setBatchSize(newBatchSize) {
  batchSize.value = newBatchSize
  setBatchSizeServer(nnNameChoice.value, batchSize.value, toaster)
}

function setLearnRate(newLearnRate) {
  learnRate.value = newLearnRate
  setLearnRateServer(nnNameChoice.value, learnRate.value, toaster)
}

function nnRestart() {
  backEnable.value = false
  nnRestartServer(cy, nnNameChoice.value, toaster)
}
</script>

<style scoped>
.trainStep {
  position: fixed;
  bottom: 50px;
  right: 0;
  width: 300px;
  height: 40px;
}

.navbar {
  flex: 0 1 auto;
}

#cy {
  flex: 1 1 auto;
}

footer {
  flex: 0 1 50px;
}
</style>
