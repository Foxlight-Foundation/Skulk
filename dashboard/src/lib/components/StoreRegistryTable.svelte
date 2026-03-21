<script lang="ts">
  import type { StoreRegistryEntry } from "$lib/stores/app.svelte";

  let {
    entries,
    loading,
    activeModelIds = [],
    onrefresh,
    oninfo,
    ondelete,
  }: {
    entries: StoreRegistryEntry[];
    loading: boolean;
    activeModelIds: string[];
    onrefresh: () => void;
    oninfo: (entry: StoreRegistryEntry) => void;
    ondelete: (entry: StoreRegistryEntry, isActive: boolean) => void;
  } = $props();

  function formatBytes(bytes: number): string {
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
  }

  function timeAgo(iso: string): string {
    const ms = Date.now() - new Date(iso).getTime();
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return "just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  }

  function isActive(modelId: string): boolean {
    return activeModelIds.includes(modelId);
  }
</script>

<div class="space-y-4">
  <div class="flex items-center justify-between">
    <div class="text-sm text-exo-light-gray">
      {entries.length} model{entries.length !== 1 ? "s" : ""} in store
    </div>
    <button
      type="button"
      class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
      onclick={onrefresh}
    >
      Refresh
    </button>
  </div>

  {#if loading}
    <div class="space-y-2">
      {#each Array(4) as _}
        <div
          class="h-12 rounded bg-exo-medium-gray/20 animate-pulse"
        ></div>
      {/each}
    </div>
  {:else if entries.length === 0}
    <div
      class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
    >
      No models in store.
    </div>
  {:else}
    <div class="rounded border border-exo-medium-gray/30 overflow-hidden">
      <table class="w-full text-sm">
        <thead>
          <tr class="bg-exo-black/40 text-exo-light-gray text-xs font-mono uppercase tracking-wider">
            <th class="text-left px-4 py-3">Model</th>
            <th class="text-right px-4 py-3">Size</th>
            <th class="text-right px-4 py-3">Files</th>
            <th class="text-right px-4 py-3">Added</th>
            <th class="w-20 px-4 py-3"></th>
          </tr>
        </thead>
        <tbody>
          {#each entries as entry}
            {@const active = isActive(entry.model_id)}
            <tr
              class="border-t border-exo-medium-gray/20 hover:bg-exo-medium-gray/10 transition-colors group"
            >
              <td class="px-4 py-3">
                <div class="flex items-center gap-2">
                  <span class="font-mono text-white">{entry.model_id}</span>
                  {#if active}
                    <span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-green-500/10 text-green-400 border border-green-500/20">
                      <span class="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse"></span>
                      active
                    </span>
                  {/if}
                </div>
              </td>
              <td class="px-4 py-3 text-right font-mono text-exo-light-gray">
                {formatBytes(entry.total_bytes)}
              </td>
              <td class="px-4 py-3 text-right font-mono text-exo-light-gray">
                {entry.files.length}
              </td>
              <td
                class="px-4 py-3 text-right font-mono text-exo-light-gray"
                title={entry.downloaded_at}
              >
                {timeAgo(entry.downloaded_at)}
              </td>
              <td class="px-4 py-3">
                <div class="flex items-center justify-end gap-1 opacity-40 group-hover:opacity-100 transition-opacity">
                  <!-- Info icon -->
                  <button
                    type="button"
                    class="p-1 rounded hover:bg-white/10 transition-colors"
                    onclick={() => oninfo(entry)}
                    title="View model details"
                  >
                    <svg class="w-4 h-4 text-white/60 hover:text-white/80" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z" />
                    </svg>
                  </button>
                  <!-- Delete icon -->
                  <button
                    type="button"
                    class="p-1 rounded hover:bg-red-500/10 transition-colors"
                    onclick={() => ondelete(entry, active)}
                    title="Delete from store"
                  >
                    <svg class="w-4 h-4 text-white/60 hover:text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2" />
                    </svg>
                  </button>
                </div>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>
