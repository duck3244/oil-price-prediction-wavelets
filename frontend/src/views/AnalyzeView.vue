<script setup lang="ts">
import { reactive } from 'vue'
import { useRouter } from 'vue-router'
import { usePredictionStore } from '@/stores/prediction'

const store = usePredictionStore()
const router = useRouter()

const form = reactive({
  symbol: 'CL=F',
  days: 30,
  wavelet: 'db4',
  decomposition_level: 4,
  sequence_length: 60,
  epochs: 30,
})

async function submit(): Promise<void> {
  await store.runPrediction({ ...form })
  if (store.result && !store.error) {
    void router.push('/')
  }
}
</script>

<template>
  <section>
    <h2 class="text-2xl font-semibold">Analyze</h2>
    <p class="mt-2 text-sm text-slate-600">
      Training runs live on every submit; expect roughly 30–90s depending on
      <code>epochs</code> and data size.
    </p>

    <form
      class="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2"
      @submit.prevent="submit"
    >
      <label class="flex flex-col gap-1 text-sm">
        <span class="font-medium text-slate-700">Symbol</span>
        <select v-model="form.symbol" class="rounded border border-slate-300 px-2 py-1">
          <option value="CL=F">WTI (CL=F)</option>
          <option value="BZ=F">Brent (BZ=F)</option>
        </select>
      </label>

      <label class="flex flex-col gap-1 text-sm">
        <span class="font-medium text-slate-700">Wavelet</span>
        <select v-model="form.wavelet" class="rounded border border-slate-300 px-2 py-1">
          <option value="db4">db4</option>
          <option value="db8">db8</option>
          <option value="haar">haar</option>
          <option value="bior2.2">bior2.2</option>
          <option value="coif3">coif3</option>
        </select>
      </label>

      <label class="flex flex-col gap-1 text-sm">
        <span class="font-medium text-slate-700">Prediction days</span>
        <input
          v-model.number="form.days"
          type="number"
          min="1"
          max="90"
          class="rounded border border-slate-300 px-2 py-1"
        />
      </label>

      <label class="flex flex-col gap-1 text-sm">
        <span class="font-medium text-slate-700">Decomposition level</span>
        <input
          v-model.number="form.decomposition_level"
          type="number"
          min="1"
          max="8"
          class="rounded border border-slate-300 px-2 py-1"
        />
      </label>

      <label class="flex flex-col gap-1 text-sm">
        <span class="font-medium text-slate-700">Sequence length</span>
        <input
          v-model.number="form.sequence_length"
          type="number"
          min="20"
          max="200"
          class="rounded border border-slate-300 px-2 py-1"
        />
      </label>

      <label class="flex flex-col gap-1 text-sm">
        <span class="font-medium text-slate-700">Epochs</span>
        <input
          v-model.number="form.epochs"
          type="number"
          min="1"
          max="300"
          class="rounded border border-slate-300 px-2 py-1"
        />
      </label>

      <div class="md:col-span-2">
        <button
          type="submit"
          :disabled="store.loading"
          class="rounded bg-sky-600 px-4 py-2 font-medium text-white hover:bg-sky-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {{ store.loading ? 'Training…' : 'Run Prediction' }}
        </button>
        <span v-if="store.error" class="ml-4 text-sm text-rose-600">
          {{ store.error }}
        </span>
      </div>
    </form>
  </section>
</template>
