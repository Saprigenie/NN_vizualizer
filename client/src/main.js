import { createApp } from 'vue'
import App from '@/App.vue'

import 'bootstrap'
import 'bootstrap-icons/font/bootstrap-icons.css'

async function bootstrap() {
  const app = createApp(App)
  app.mount('#app')
}

bootstrap()
