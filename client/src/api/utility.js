// Получает максимум из массивов разных вложенностей.
export function getMax(arr) {
  return Math.max(...arr.map((elem) => (Array.isArray(elem) ? getMax(elem) : elem)))
}
