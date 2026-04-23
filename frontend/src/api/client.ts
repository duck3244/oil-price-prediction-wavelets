import axios from 'axios'

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api',
  timeout: 600_000,
})

export interface PredictRequest {
  symbol?: string
  days?: number
  wavelet?: string
  decomposition_level?: number
  sequence_length?: number
  epochs?: number
}

export interface ComponentPrediction {
  name: string
  values: number[]
}

export interface PredictResponse {
  symbol: string
  current_price: number
  historical_dates: string[]
  historical_prices: number[]
  predictions: number[]
  component_predictions: ComponentPrediction[]
  wavelet: string
  decomposition_level: number
  generated_at: string
}

export interface HealthResponse {
  status: string
  tensorflow_version: string
  gpu_available: boolean
}

export async function predict(req: PredictRequest): Promise<PredictResponse> {
  const { data } = await apiClient.post<PredictResponse>('/predict', req)
  return data
}

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await apiClient.get<HealthResponse>('/health')
  return data
}
