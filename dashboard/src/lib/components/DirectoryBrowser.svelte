<script lang="ts">
  import {
    browseFilesystem,
    type DirectoryEntry,
  } from "$lib/stores/app.svelte";

  let {
    value = $bindable(""),
    onselect,
    label = "Path",
  }: {
    value: string;
    onselect: (path: string) => void;
    label?: string;
  } = $props();

  let open = $state(false);
  let browsePath = $state("/Volumes");
  let directories = $state<DirectoryEntry[]>([]);
  let loading = $state(false);
  let error = $state<string | null>(null);

  async function browse(path: string) {
    loading = true;
    error = null;
    try {
      const result = await browseFilesystem(path);
      browsePath = result.path;
      directories = result.directories;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      directories = [];
    } finally {
      loading = false;
    }
  }

  function handleOpen() {
    open = !open;
    if (open) {
      browse(value || "/Volumes");
    }
  }

  function navigateTo(path: string) {
    browse(path);
  }

  function selectCurrent() {
    value = browsePath;
    onselect(browsePath);
    open = false;
  }

  // Breadcrumb segments from browsePath
  const segments = $derived.by(() => {
    const parts = browsePath.split("/").filter(Boolean);
    const result: { name: string; path: string }[] = [];
    let acc = "";
    for (const part of parts) {
      acc += "/" + part;
      result.push({ name: part, path: acc });
    }
    return result;
  });
</script>

<div class="space-y-1">
  <label class="text-xs text-exo-light-gray uppercase tracking-wide"
    >{label}</label
  >
  <div class="flex gap-2">
    <input
      type="text"
      bind:value
      placeholder="/Volumes/ModelStore/models"
      class="flex-1 bg-exo-dark-gray border border-exo-medium-gray/40 rounded px-3 py-2 text-sm font-mono text-white placeholder:text-exo-light-gray/30 focus:border-exo-yellow focus:outline-none"
    />
    <button
      type="button"
      class="px-3 py-2 border border-exo-medium-gray/40 rounded text-exo-light-gray hover:text-exo-yellow hover:border-exo-yellow/40 transition-colors"
      onclick={handleOpen}
      title="Browse filesystem"
    >
      <svg
        class="w-4 h-4"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
        stroke-linecap="round"
        stroke-linejoin="round"
      >
        <path
          d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z"
        />
      </svg>
    </button>
  </div>

  {#if open}
    <div
      class="rounded border border-exo-medium-gray/40 bg-exo-black/60 overflow-hidden"
    >
      <!-- Breadcrumb -->
      <div
        class="flex items-center gap-1 px-3 py-2 bg-exo-black/40 border-b border-exo-medium-gray/20 text-xs font-mono overflow-x-auto"
      >
        <button
          type="button"
          class="text-exo-light-gray hover:text-exo-yellow transition-colors shrink-0"
          onclick={() => navigateTo("/Volumes")}>/</button
        >
        {#each segments as seg, i}
          {#if i > 0}
            <span class="text-exo-medium-gray shrink-0">/</span>
          {/if}
          <button
            type="button"
            class="text-exo-light-gray hover:text-exo-yellow transition-colors shrink-0"
            onclick={() => navigateTo(seg.path)}>{seg.name}</button
          >
        {/each}
      </div>

      <!-- Directory listing -->
      <div class="max-h-48 overflow-y-auto">
        {#if loading}
          <div class="px-3 py-4 text-center text-xs text-exo-light-gray">
            Loading...
          </div>
        {:else if error}
          <div class="px-3 py-4 text-center text-xs text-red-400">
            {error}
          </div>
        {:else if directories.length === 0}
          <div class="px-3 py-4 text-center text-xs text-exo-light-gray">
            Empty directory
          </div>
        {:else}
          {#each directories as dir}
            <button
              type="button"
              class="w-full flex items-center gap-2 px-3 py-1.5 text-sm font-mono text-white/80 hover:bg-exo-medium-gray/20 hover:text-white transition-colors text-left"
              onclick={() => navigateTo(dir.path)}
            >
              <svg
                class="w-3.5 h-3.5 text-exo-yellow/60 shrink-0"
                viewBox="0 0 24 24"
                fill="currentColor"
              >
                <path
                  d="M10 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V8a2 2 0 00-2-2h-8l-2-2z"
                />
              </svg>
              {dir.name}
            </button>
          {/each}
        {/if}
      </div>

      <!-- Select button -->
      <div class="px-3 py-2 border-t border-exo-medium-gray/20">
        <button
          type="button"
          class="w-full px-3 py-1.5 text-xs font-mono uppercase tracking-wider bg-exo-yellow text-exo-black rounded hover:bg-exo-yellow/90 transition-colors"
          onclick={selectCurrent}
        >
          Select this directory
        </button>
      </div>
    </div>
  {/if}
</div>
