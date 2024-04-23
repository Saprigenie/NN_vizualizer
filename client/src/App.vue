<template>
  <nav class="navbar navbar-expand navbar-dark bg-dark">
    <router-link class="navbar-brand" to="/">
      <img
        src="./assets/logo.ico"
        alt="NN визуализатор обучения"
        width="40"
        height="40"
        class="ps-2 d-inline-block"
      />
    </router-link>

    <div class="collapse navbar-collapse" id="navbarSupportedContent">
      <ul class="navbar-nav mr-auto">
        <li class="nav-item">
          <router-link class="nav-link" to="/"> Главная </router-link>
        </li>

        <li class="nav-item">
          <router-link class="nav-link" to="/help"> Справка </router-link>
        </li>
      </ul>
    </div>
  </nav>

  <main>
    <router-view></router-view>
    <ToasterComponent />
  </main>
</template>

<script setup>
import ToasterComponent from '@/components/utility/ToasterComponent.vue'
</script>

<style scoped>
nav {
  flex: 0 1 auto;
}

main {
  display: flex;
  flex-flow: column;
  flex: 1 1 auto;
}
</style>
