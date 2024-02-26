import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    component: () => import('@/components/pages/Main.vue')
  },
  { path: '/help', component: () => import('@/components/pages/Help.vue') }
]

export const router = createRouter({
  history: createWebHashHistory(),
  routes
})
