<template>
  <div ref="toastElement" class="toast" role="alert">
    <div class="toast-header">
      <slot name="title"></slot>
    </div>
    <div class="toast-body">
      <slot name="body"></slot>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, readonly } from 'vue'
import { useEventListener } from '@vueuse/core'
import { Toast } from 'bootstrap'

const toastElement = ref(null)
const toast = ref(null)

const props = defineProps({
  animation: {
    type: Boolean,
    default: true
  },
  autohide: {
    type: Boolean,
    default: true
  },
  delay: {
    type: Number,
    default: 3500
  }
})

console.log(props)

const emit = defineEmits(['hide', 'hidden', 'show', 'shown', 'mounted'])

onMounted(() => {
  if (toastElement.value)
    toast.value = new Toast(toastElement.value, {
      delay: props.delay,
      animation: props.animation,
      autohide: props.autohide
    })

  useEventListener(toastElement, 'hide.bs.toast', () => {
    emit('hide')
  })
  useEventListener(toastElement, 'hidden.bs.toast', () => {
    emit('hidden')
  })
  useEventListener(toastElement, 'show.bs.toast', () => {
    emit('show')
  })
  useEventListener(toastElement, 'shown.bs.toast', () => {
    emit('shown')
  })

  emit('mounted', toast.value)
})

defineExpose({
  toast: readonly(toast)
})
</script>

<style scoped lang="scss"></style>
