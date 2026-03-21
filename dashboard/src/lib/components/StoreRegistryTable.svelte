<script lang="ts">
  import type { StoreRegistryEntry } from "$lib/stores/app.svelte";

  let {
    entries,
    loading,
    onrefresh,
  }: {
    entries: StoreRegistryEntry[];
    loading: boolean;
    onrefresh: () => void;
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
          </tr>
        </thead>
        <tbody>
          {#each entries as entry}
            <tr
              class="border-t border-exo-medium-gray/20 hover:bg-exo-medium-gray/10 transition-colors"
            >
              <td class="px-4 py-3 font-mono text-white">
                {entry.model_id}
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
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>
