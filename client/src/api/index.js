import axios from 'axios'

const host = import.meta.env.CLIENT_SERVER_URL ?? 'localhost'
const port = import.meta.env.CLIENT_SERVER_PORT ?? '5000'

export const serverURL = `http://${host}:${port}`
export const baseURL = `${serverURL}/api`

export const api = axios.create({
  baseURL,
  withCredentials: true
})
