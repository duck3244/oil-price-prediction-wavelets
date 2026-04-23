<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import PriceChart from '@/components/PriceChart.vue'
import { usePredictionStore } from '@/stores/prediction'

const store = usePredictionStore()

interface Summary {
  current: number
  nextDay: number
  change: number
  week?: number
  weekChange?: number
}

const summary = computed<Summary | null>(() => {
  const r = store.result
  if (!r || r.predictions.length === 0) return null
  const current = r.current_price
  const nextDay = r.predictions[0]!
  const change = ((nextDay - current) / current) * 100
  const week = r.predictions[6]
  const weekChange = week !== undefined ? ((week - current) / current) * 100 : undefined
  return { current, nextDay, change, week, weekChange }
})
</script>

<template>
  <section>
    <h2 class="text-2xl font-semibold">Dashboard</h2>
    <p class="mt-2 text-sm text-slate-600">
      Run an analysis from the
      <RouterLink to="/analyze" class="text-sky-700 underline">Analyze</RouterLink>
      page to populate this view.
    </p>

    <div v-if="!store.result" class="mt-8 rounded-lg border border-dashed bg-white p-8 text-center">
      <p class="text-sm text-slate-500">No results yet.</p>
    </div>

    <template v-else-if="summary">
      <div class="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div class="rounded-lg border bg-white p-4">
          <div class="text-xs uppercase tracking-wide text-slate-500">Current</div>
          <div class="mt-1 text-2xl font-semibold">${{ summary.current.toFixed(2) }}</div>
        </div>
        <div class="rounded-lg border bg-white p-4">
          <div class="text-xs uppercase tracking-wide text-slate-500">Next Day</div>
          <div class="mt-1 text-2xl font-semibold">${{ summary.nextDay.toFixed(2) }}</div>
          <div
            class="text-sm"
            :class="summary.change >= 0 ? 'text-emerald-600' : 'text-rose-600'"
          >
            {{ summary.change >= 0 ? '+' : '' }}{{ summary.change.toFixed(2) }}%
          </div>
        </div>
        <div v-if="summary.week !== undefined" class="rounded-lg border bg-white p-4">
          <div class="text-xs uppercase tracking-wide text-slate-500">1 Week</div>
          <div class="mt-1 text-2xl font-semibold">${{ summary.week.toFixed(2) }}</div>
          <div
            v-if="summary.weekChange !== undefined"
            class="text-sm"
            :class="summary.weekChange >= 0 ? 'text-emerald-600' : 'text-rose-600'"
          >
            {{ summary.weekChange >= 0 ? '+' : '' }}{{ summary.weekChange.toFixed(2) }}%
          </div>
        </div>
      </div>

      <div class="mt-8 rounded-lg border bg-white p-4">
        <h3 class="mb-4 text-sm font-semibold text-slate-700">
          {{ store.result.symbol }} — historical + {{ store.result.predictions.length }}-day forecast
          <span class="ml-2 text-xs font-normal text-slate-500"
            >wavelet: {{ store.result.wavelet }}, L={{ store.result.decomposition_level }}</span
          >
        </h3>
        <PriceChart
          :historical-dates="store.result.historical_dates"
          :historical-prices="store.result.historical_prices"
          :predictions="store.result.predictions"
        />
      </div>
    </template>
  </section>
</template>
