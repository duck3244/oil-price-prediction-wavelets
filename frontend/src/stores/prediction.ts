import { defineStore } from 'pinia'
import { ref } from 'vue'
import { predict, type PredictRequest, type PredictResponse } from '@/api/client'

export const usePredictionStore = defineStore('prediction', () => {
  const result = ref<PredictResponse | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function runPrediction(req: PredictRequest): Promise<void> {
    loading.value = true
    error.value = null
    try {
      result.value = await predict(req)
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'unknown error'
    } finally {
      loading.value = false
    }
  }

  function reset(): void {
    result.value = null
    error.value = null
  }

  return { result, loading, error, runPrediction, reset }
})
