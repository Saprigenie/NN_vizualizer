<template>
  <!-- <nav class="navbar navbar-expand-lg bg-light">
    <div class="container-fluid px-3">
      <router-link :to="None" class="navbar-brand">
        <i class="bi bi-map fs-2" />
      </router-link>
      <button
        class="navbar-toggler"
        type="button"
        data-bs-toggle="collapse"
        data-bs-target="#navbar"
      >
        <span class="navbar-toggler-icon"></span>
      </button>
      <div id="navbar" class="collapse navbar-collapse fs-6">
        <ul class="navbar-nav me-auto mb-2 mb-lg-0">
          <li class="nav-item ps-4">
            <router-link :to="None" class="nav-link" active-class="active"> Главная </router-link>
          </li>
          <li class="nav-item ps-4">
            <router-link :to="None" class="nav-link" active-class="text-info">
              О проекте
            </router-link>
          </li>

          <template v-if="userStore.role === UserRole.admin">
            <li class="nav-item ps-4">
              <router-link :to="None" class="nav-link text-warning" active-class="fw-bold">
                Пользователи
              </router-link>
            </li>

            <li class="nav-item ps-4">
              <router-link :to="None" class="nav-link text-danger" active-class="fw-bold">
                Дампы
              </router-link>
            </li>
          </template>
        </ul>
      </div>
      <div class="fs-2 text-primary" role="button">
        <i v-if="userStore.isAuthed" class="bi bi-box-arrow-left" @click="logout" />
        <router-link
          v-else
          :to="{ name: routeNames.Auth }"
          class="nav-link"
          active-class="text-info"
        >
          <i class="bi bi-box-arrow-in-right" />
        </router-link>
      </div>
    </div>
  </nav> -->
  <main>
    <div>
      <p>Я живой</p>
    </div>
  </main>
</template>

<script setup></script>

<style lang="scss">
@import '@/components/scss/app.scss';
</style>
