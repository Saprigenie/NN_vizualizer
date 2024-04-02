<template>
  <div class="toast-container position-fixed bottom-0 end-0 p-3">
    <BToast
      v-for="toast of toaster.toasts"
      :key="toast.id"
      :delay="toast.time"
      :class="`text-bg-${toast.type ?? 'light'}`"
      class="toast"
      role="alert"
      @hidden="toastHidden(toast)"
      @mounted="toastMounted"
    >
      <template #title>
        <span class="me-auto">{{ toast.title }}</span>
        <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close" />
      </template>

      <template #body>
        {{ toast.body }}
      </template>
    </BToast>
  </div>
</template>

<script setup>
import { useToaster } from '@/store/toaster'
import BToast from '@/bootstrap/BToast.vue'

const toaster = useToaster()

function toastMounted(toast) {
  toast.show()
}

function toastHidden(toast) {
  toaster.deleteToast(toast.id)
}
</script>

<style scoped lang="scss"></style>
