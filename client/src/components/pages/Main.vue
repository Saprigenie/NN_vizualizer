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
            <a class="dropdown-item nav-link"> ANN (Полносвязная нейронная сеть) </a>
          </li>

          <li>
            <a class="dropdown-item nav-link"> CNN (Сверточная нейронная сеть) </a>
          </li>

          <li>
            <a class="dropdown-item nav-link"> GAN (Генеративная нейронная сеть) </a>
          </li>
        </ul>
      </li>

      <div class="vr ms-2 me-2"></div>

      <li class="nav-item">
        <button class="nav-link">Кнопка</button>
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
        <button class="btn btn-dark"><i class="bi bi-arrow-left text-light"></i></button>
        <button class="btn btn-dark"><i class="bi bi-arrow-right text-light"></i></button>
      </div>
      <div class="col-sm">
        <button class="btn btn-dark float-end">
          <i class="bi bi-arrow-clockwise"></i>
        </button>
      </div>
    </div>
  </footer>
</template>

<script setup>
import cytoscape from 'cytoscape'
import { onMounted } from 'vue'

onMounted(() => {
  var cy = cytoscape({
    container: document.getElementById('cy'), // container to render in

    elements: [
      // list of graph elements to start with
      {
        // node a
        data: { id: 'a' }
      },
      {
        // node b
        data: { id: 'b' }
      },
      {
        // edge ab
        data: { id: 'ab', source: 'a', target: 'b' }
      }
    ],

    style: [
      // the stylesheet for the graph
      {
        selector: 'node',
        style: {
          'background-color': '#666',
          label: 'data(id)'
        }
      },

      {
        selector: 'edge',
        style: {
          width: 3,
          'line-color': '#ccc',
          'target-arrow-color': '#ccc',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier'
        }
      }
    ],

    layout: {
      name: 'grid',
      rows: 1
    }
  })
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
