import { defineStore } from 'pinia'
import { readonly, ref } from 'vue'

export const useToaster = defineStore('toasts', () => {
  const id = ref(0)
  const toasts = ref([])

  function addToast(toast) {
    id.value++
    toasts.value.push({ ...toast, id: id.value })
  }

  function deleteToast(id) {
    toasts.value = toasts.value.filter((toast) => toast.id !== id)
  }

  return { addToast, deleteToast, toasts: readonly(toasts) }
})

export const ToastTypes = {
  primary: 'primary',
  secondary: 'secondary',
  success: 'success',
  info: 'info',
  danger: 'danger',
  warning: 'warning',
  light: 'light',
  dark: 'dark'
}
