import { createRouter, createWebHashHistory } from 'vue-router'

// Теst components
const Home = { template: '<div>Home</div>' }
const Help = { template: '<div>About</div>' }

const routes = [
  {
    path: '/',
    component: () => import('@/components/pages/Home.vue')
  },
  { path: '/help', component: () => import('@/components/pages/Help.vue') }
]

export const router = createRouter({
  history: createWebHashHistory(),
  routes
})
