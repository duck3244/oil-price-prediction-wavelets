<script setup lang="ts">
import { computed } from 'vue'
import { Line } from 'vue-chartjs'
import {
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
} from 'chart.js'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend)

const props = defineProps<{
  historicalDates: string[]
  historicalPrices: number[]
  predictions: number[]
}>()

const chartData = computed(() => {
  const histLabels = props.historicalDates
  const predLabels = props.predictions.map((_, i) => `+${i + 1}d`)
  const labels = [...histLabels, ...predLabels]

  const histSeries = [
    ...props.historicalPrices,
    ...new Array<number | null>(props.predictions.length).fill(null),
  ]
  const predSeries = [
    ...new Array<number | null>(props.historicalPrices.length).fill(null),
    ...props.predictions,
  ]

  return {
    labels,
    datasets: [
      {
        label: 'Historical',
        data: histSeries,
        borderColor: '#0369a1',
        backgroundColor: 'rgba(3, 105, 161, 0.15)',
        pointRadius: 0,
        tension: 0.1,
        spanGaps: false,
      },
      {
        label: 'Forecast',
        data: predSeries,
        borderColor: '#dc2626',
        backgroundColor: 'rgba(220, 38, 38, 0.15)',
        borderDash: [4, 4],
        pointRadius: 0,
        tension: 0.1,
        spanGaps: false,
      },
    ],
  }
})

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index' as const, intersect: false },
  plugins: {
    legend: { position: 'top' as const },
    tooltip: { enabled: true },
  },
  scales: {
    x: { ticks: { maxTicksLimit: 10 } },
    y: { ticks: { callback: (v: string | number) => `$${v}` } },
  },
}
</script>

<template>
  <div class="h-96">
    <Line :data="chartData" :options="chartOptions" />
  </div>
</template>
