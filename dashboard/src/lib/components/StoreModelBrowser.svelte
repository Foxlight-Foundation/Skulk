<script lang="ts">
  import { requestStoreDownload } from "$lib/stores/app.svelte";

  let {
    isOpen = $bindable(false),
    storeModelIds = new Set<string>(),
    ondownloadstarted,
  }: {
    isOpen: boolean;
    storeModelIds: Set<string>;
    ondownloadstarted: () => void;
  } = $props();

  type ModelInfo = {
    id: string;
    name?: string;
    storage_size_megabytes?: number;
    base_model?: string;
    quantization?: string;
    family?: string;
  };

  let query = $state("");
  let models = $state<ModelInfo[]>([]);
  let searchResults = $state<ModelInfo[]>([]);
  let loading = $state(false);
  let searching = $state(false);
  let downloadingIds = $state(new Set<string>());
  let searchTimeout: ReturnType<typeof setTimeout> | null = null;

  async function fetchModels() {
    loading = true;
    try {
      const resp = await fetch("/models");
      if (resp.ok) {
        const data = await resp.json();
        models = data.data || [];
      }
    } catch { /* ignore */ }
    loading = false;
  }

  async function searchHuggingFace(q: string) {
    if (!q.trim()) {
      searchResults = [];
      return;
    }
    searching = true;
    try {
      const resp = await fetch(`/models/search?query=${encodeURIComponent(q)}&limit=20`);
      if (resp.ok) {
        const data = await resp.json();
        searchResults = (data.data || data || []).map((r: Record<string, unknown>) => ({
          id: r.id || r.modelId || "",
          name: (r.name as string) || (r.id as string) || "",
        }));
      }
    } catch { /* ignore */ }
    searching = false;
  }

  function handleSearch(value: string) {
    query = value;
    if (searchTimeout) clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => searchHuggingFace(value), 300);
  }

  async function handleDownload(modelId: string) {
    downloadingIds = new Set([...downloadingIds, modelId]);
    try {
      await requestStoreDownload(modelId);
      ondownloadstarted();
    } catch (err) {
      console.error("Store download failed:", err);
    }
  }

  function formatSize(mb: number | undefined): string {
    if (!mb) return "";
    if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb} MB`;
  }

  const filteredModels = $derived.by(() => {
    const q = query.toLowerCase().trim();
    if (!q) return models;
    return models.filter(
      (m) =>
        m.id.toLowerCase().includes(q) ||
        (m.name && m.name.toLowerCase().includes(q)) ||
        (m.family && m.family.toLowerCase().includes(q)),
    );
  });

  const displayModels = $derived(
    query.trim() && searchResults.length > 0 ? searchResults : filteredModels,
  );

  const isSearching = $derived(query.trim().length > 0 && searching);

  $effect(() => {
    if (isOpen) fetchModels();
  });
</script>

{#if isOpen}
  <div
    class="fixed inset-0 z-[60] bg-black/60"
    onclick={() => (isOpen = false)}
    role="presentation"
  ></div>
  <div
    class="fixed z-[60] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(95vw,600px)] max-h-[80vh] bg-exo-dark-gray border border-exo-yellow/10 rounded-lg shadow-2xl flex flex-col"
    role="dialog"
    aria-modal="true"
  >
    <!-- Header -->
    <div class="flex items-center justify-between px-4 py-3 border-b border-exo-medium-gray/20">
      <h3 class="font-mono text-lg text-white">Find Models</h3>
      <button
        type="button"
        class="p-1 rounded hover:bg-white/10 transition-colors text-white/50"
        onclick={() => (isOpen = false)}
      >
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
          <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z" />
        </svg>
      </button>
    </div>

    <!-- Search -->
    <div class="px-4 py-3 border-b border-exo-medium-gray/20">
      <input
        type="text"
        placeholder="Search models or HuggingFace..."
        value={query}
        oninput={(e) => handleSearch(e.currentTarget.value)}
        class="w-full bg-exo-black/40 border border-exo-medium-gray/40 rounded px-3 py-2 text-sm font-mono text-white placeholder:text-exo-light-gray/30 focus:border-exo-yellow focus:outline-none"
      />
      {#if isSearching}
        <div class="text-xs text-exo-light-gray/50 mt-1 font-mono">Searching HuggingFace...</div>
      {/if}
    </div>

    <!-- Model list -->
    <div class="flex-1 overflow-y-auto">
      {#if loading}
        <div class="px-4 py-8 text-center text-sm text-exo-light-gray">Loading models...</div>
      {:else if displayModels.length === 0}
        <div class="px-4 py-8 text-center text-sm text-exo-light-gray">
          {query.trim() ? "No models found" : "No models available"}
        </div>
      {:else}
        {#each displayModels as model}
          {@const inStore = storeModelIds.has(model.id)}
          {@const isDownloading = downloadingIds.has(model.id)}
          <div class="flex items-center gap-3 px-4 py-2.5 border-b border-exo-medium-gray/10 hover:bg-exo-medium-gray/10 transition-colors group">
            <div class="flex-1 min-w-0">
              <div class="text-sm font-mono text-white truncate">{model.id}</div>
              <div class="flex items-center gap-2 text-[11px] text-exo-light-gray/60">
                {#if model.family}
                  <span>{model.family}</span>
                {/if}
                {#if model.quantization}
                  <span>{model.quantization}</span>
                {/if}
                {#if model.storage_size_megabytes}
                  <span>{formatSize(model.storage_size_megabytes)}</span>
                {/if}
              </div>
            </div>
            {#if inStore}
              <span class="text-[10px] font-mono uppercase text-green-400/70 border border-green-500/20 rounded px-1.5 py-0.5 shrink-0">
                In Store
              </span>
            {:else if isDownloading}
              <span class="text-[10px] font-mono uppercase text-exo-yellow/70 border border-exo-yellow/20 rounded px-1.5 py-0.5 shrink-0 animate-pulse">
                Downloading...
              </span>
            {:else}
              <button
                type="button"
                class="p-1.5 rounded hover:bg-exo-yellow/10 transition-colors opacity-40 group-hover:opacity-100 shrink-0"
                onclick={() => handleDownload(model.id)}
                title="Download to store"
              >
                <svg class="w-4 h-4 text-exo-yellow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M12 3v12" />
                  <path d="M7 12l5 5 5-5" />
                  <path d="M5 21h14" />
                </svg>
              </button>
            {/if}
          </div>
        {/each}
      {/if}
    </div>
  </div>
{/if}
