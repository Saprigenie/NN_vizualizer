import axios from 'axios'

import { useToaster, ToastTypes } from '@/store/toaster'

const host = import.meta.env.CLIENT_SERVER_URL ?? 'localhost'
const port = import.meta.env.CLIENT_SERVER_PORT ?? '5000'

export const serverURL = `http://${host}:${port}`
export const baseURL = `${serverURL}/api`

export const api = axios.create({
  baseURL,
  withCredentials: true
})

api.interceptors.response.use(
  async (res) => {
    return res
  },
  async (error) => {
    const toaster = useToaster()

    let data
    if (error?.response?.data?.constructor === ArrayBuffer) {
      try {
        data = JSON.parse(new TextDecoder().decode(error?.response?.data))
      } catch {
        data = {}
      }
    } else {
      data = error?.response?.data
    }

    console.error(error?.response)
    toaster.addToast({
      title: 'Произошла ошибка',
      body:
        data?.message ??
        data ??
        'Cервер не может обработать запрос в данный момент, попробуйте позднее.',
      type: ToastTypes.danger
    })

    throw error
  }
)
