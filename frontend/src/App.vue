<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="min-h-screen bg-slate-50 text-slate-900">
    <header class="border-b bg-white">
      <nav class="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
        <h1 class="text-lg font-semibold">Oil Predictor</h1>
        <RouterLink
          to="/"
          class="text-sm text-slate-600 hover:text-sky-700"
          active-class="text-sky-700 font-medium"
          >Dashboard</RouterLink
        >
        <RouterLink
          to="/analyze"
          class="text-sm text-slate-600 hover:text-sky-700"
          active-class="text-sky-700 font-medium"
          >Analyze</RouterLink
        >
      </nav>
    </header>
    <main class="mx-auto max-w-6xl px-6 py-8">
      <RouterView />
    </main>
  </div>
</template>
