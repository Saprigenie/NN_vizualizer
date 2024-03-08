import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    component: () => import('@/components/pages/main/Main.vue')
  },
  { path: '/help', component: () => import('@/components/pages/help/Help.vue') }
]

export const router = createRouter({
  history: createWebHashHistory(),
  routes
})
