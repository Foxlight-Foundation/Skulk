<script lang="ts">
  import { onMount } from "svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import {
    fetchConfig,
    updateConfig,
    fetchStoreHealth,
    fetchNodeIdentity,
    type ConfigResponse,
    type StoreHealthResponse,
    type NodeIdentityResponse,
  } from "$lib/stores/app.svelte";
  import DirectoryBrowser from "$lib/components/DirectoryBrowser.svelte";

  let loading = $state(true);
  let saving = $state(false);
  let saveMessage = $state<string | null>(null);
  let saveError = $state<string | null>(null);
  let storeHealth = $state<StoreHealthResponse | null>(null);
  let fileExists = $state(false);
  let configPath = $state("");

  // Node identity (fetched on mount)
  let nodeIdentity = $state<NodeIdentityResponse | null>(null);
  let isThisNodeStoreHost = $state(false);

  // Form state — model_store section
  let enabled = $state(true);
  let storeHost = $state("");
  let storeHttpHost = $state("");
  let storePort = $state(58080);
  let storePath = $state("");
  let allowHfFallback = $state(true);
  let stagingEnabled = $state(true);
  let nodeCachePath = $state("~/.exo/staging");
  let cleanupOnDeactivate = $state(true);

  // Node overrides as editable list
  let overrides = $state<
    { key: string; stagingEnabled: boolean; nodeCachePath: string; cleanupOnDeactivate: boolean }[]
  >([]);

  function loadConfig(data: ConfigResponse) {
    fileExists = data.fileExists;
    configPath = data.configPath;
    const ms = data.config?.model_store as Record<string, unknown> | undefined;
    if (!ms) return;
    enabled = (ms.enabled as boolean) ?? true;
    storeHost = (ms.store_host as string) ?? "";
    storeHttpHost = (ms.store_http_host as string) ?? "";
    storePort = (ms.store_port as number) ?? 58080;
    storePath = (ms.store_path as string) ?? "";
    // Detect if this node is the configured store host
    if (nodeIdentity && storeHost) {
      isThisNodeStoreHost =
        storeHost === nodeIdentity.hostname ||
        storeHost === nodeIdentity.nodeId;
    }
    const dl = ms.download as Record<string, unknown> | undefined;
    allowHfFallback = (dl?.allow_hf_fallback as boolean) ?? true;
    const stg = ms.staging as Record<string, unknown> | undefined;
    stagingEnabled = (stg?.enabled as boolean) ?? true;
    nodeCachePath = (stg?.node_cache_path as string) ?? "~/.exo/staging";
    cleanupOnDeactivate = (stg?.cleanup_on_deactivate as boolean) ?? true;
    const no = ms.node_overrides as Record<
      string,
      Record<string, unknown>
    > | undefined;
    if (no) {
      overrides = Object.entries(no).map(([key, val]) => {
        const s = val.staging as Record<string, unknown> | undefined;
        return {
          key,
          stagingEnabled: (s?.enabled as boolean) ?? true,
          nodeCachePath: (s?.node_cache_path as string) ?? "",
          cleanupOnDeactivate: (s?.cleanup_on_deactivate as boolean) ?? true,
        };
      });
    }
  }

  function buildConfig(): Record<string, unknown> {
    const nodeOverrides: Record<string, unknown> = {};
    for (const o of overrides) {
      if (o.key.trim()) {
        nodeOverrides[o.key.trim()] = {
          staging: {
            enabled: o.stagingEnabled,
            node_cache_path: o.nodeCachePath,
            cleanup_on_deactivate: o.cleanupOnDeactivate,
          },
        };
      }
    }
    return {
      model_store: {
        enabled,
        store_host: storeHost,
        ...(storeHttpHost ? { store_http_host: storeHttpHost } : {}),
        store_port: storePort,
        store_path: storePath,
        download: { allow_hf_fallback: allowHfFallback },
        staging: {
          enabled: stagingEnabled,
          node_cache_path: nodeCachePath,
          cleanup_on_deactivate: cleanupOnDeactivate,
        },
        ...(Object.keys(nodeOverrides).length > 0
          ? { node_overrides: nodeOverrides }
          : {}),
      },
    };
  }

  const canSave = $derived(
    !enabled || (storeHost.trim() !== "" && storePath.trim() !== ""),
  );

  async function handleSave() {
    if (!canSave) return;
    saving = true;
    saveMessage = null;
    saveError = null;
    try {
      const result = await updateConfig(buildConfig());
      saveMessage = result.message;
    } catch (err) {
      saveError = err instanceof Error ? err.message : String(err);
    } finally {
      saving = false;
    }
  }

  function addOverride() {
    overrides = [
      ...overrides,
      { key: "", stagingEnabled: true, nodeCachePath: "~/.exo/staging", cleanupOnDeactivate: true },
    ];
  }

  function removeOverride(index: number) {
    overrides = overrides.filter((_, i) => i !== index);
  }

  function formatBytes(bytes: number): string {
    if (bytes === 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
  }

  function handleStoreHostToggle(isHost: boolean) {
    isThisNodeStoreHost = isHost;
    if (isHost && nodeIdentity) {
      storeHost = nodeIdentity.hostname;
      storeHttpHost = nodeIdentity.ipAddress;
    } else {
      storeHost = "";
      storeHttpHost = "";
    }
  }

  onMount(async () => {
    try {
      const [configData, health, identity] = await Promise.all([
        fetchConfig(),
        fetchStoreHealth(),
        fetchNodeIdentity(),
      ]);
      nodeIdentity = identity;
      loadConfig(configData);
      storeHealth = health;
    } catch (err) {
      console.error("Failed to load settings:", err);
    } finally {
      loading = false;
    }
  });
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-3xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div>
      <h1
        class="text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow"
      >
        Settings
      </h1>
      <p class="text-sm text-exo-light-gray">
        Model store configuration
        {#if configPath}
          <span class="text-exo-light-gray/50">— {configPath}</span>
        {/if}
      </p>
    </div>

    {#if loading}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-8 text-center text-exo-light-gray"
      >
        Loading…
      </div>
    {:else}
      <!-- Store Health -->
      <div class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-3">
        <h2 class="text-xs font-mono uppercase tracking-widest text-exo-light-gray">
          Store Status
        </h2>
        {#if storeHealth}
          <div class="flex items-center gap-3">
            <div class="w-2 h-2 rounded-full bg-green-400"></div>
            <span class="text-sm font-mono">Connected</span>
          </div>
          <div class="text-sm text-exo-light-gray space-y-1">
            <div>
              Path: <span class="font-mono text-white">{storeHealth.storePath}</span>
            </div>
            <div class="flex items-center gap-3">
              <span>Disk:</span>
              <div class="flex-1 h-2 rounded bg-exo-medium-gray/40 overflow-hidden max-w-xs">
                <div
                  class="h-full bg-exo-yellow rounded"
                  style="width: {((storeHealth.usedBytes / storeHealth.totalBytes) * 100).toFixed(1)}%"
                ></div>
              </div>
              <span class="font-mono text-white text-xs">
                {formatBytes(storeHealth.usedBytes)} / {formatBytes(storeHealth.totalBytes)}
              </span>
            </div>
          </div>
        {:else}
          <div class="flex items-center gap-3">
            <div class="w-2 h-2 rounded-full bg-exo-medium-gray"></div>
            <span class="text-sm text-exo-light-gray">
              {enabled ? "Store unreachable" : "Store not enabled"}
            </span>
          </div>
        {/if}
      </div>

      <!-- Config Form -->
      <form
        class="space-y-6"
        onsubmit={(e) => {
          e.preventDefault();
          handleSave();
        }}
      >
        <!-- Model Store -->
        <fieldset class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-4">
          <legend class="text-xs font-mono uppercase tracking-widest text-exo-light-gray px-2">
            Model Store
          </legend>
          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              bind:checked={enabled}
              class="w-4 h-4 accent-exo-yellow"
            />
            <span class="text-sm">Enabled</span>
          </label>
          <!-- Store host toggle -->
          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={isThisNodeStoreHost}
              onchange={(e) => handleStoreHostToggle(e.currentTarget.checked)}
              class="w-4 h-4 accent-exo-yellow"
            />
            <span class="text-sm">This node is the store host</span>
          </label>
          {#if isThisNodeStoreHost && nodeIdentity}
            <div class="text-sm text-exo-light-gray bg-exo-black/40 rounded px-3 py-2 font-mono">
              Serving as <span class="text-white">{nodeIdentity.hostname}</span>
              <span class="text-exo-light-gray/50">({nodeIdentity.ipAddress})</span>
            </div>
          {:else}
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div class="space-y-1">
                <label for="store-host" class="text-xs text-exo-light-gray uppercase tracking-wide">Store Host</label>
                <input
                  id="store-host"
                  type="text"
                  bind:value={storeHost}
                  placeholder="e.g. mac-studio-1"
                  class="w-full bg-exo-dark-gray border border-exo-medium-gray/40 rounded px-3 py-2 text-sm font-mono text-white placeholder:text-exo-light-gray/30 focus:border-exo-yellow focus:outline-none"
                />
              </div>
              <div class="space-y-1">
                <label for="store-http-host" class="text-xs text-exo-light-gray uppercase tracking-wide">HTTP Host <span class="text-exo-light-gray/50">(optional)</span></label>
                <input
                  id="store-http-host"
                  type="text"
                  bind:value={storeHttpHost}
                  placeholder="defaults to store host"
                  class="w-full bg-exo-dark-gray border border-exo-medium-gray/40 rounded px-3 py-2 text-sm font-mono text-white placeholder:text-exo-light-gray/30 focus:border-exo-yellow focus:outline-none"
                />
              </div>
            </div>
          {/if}
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div class="space-y-1">
              <label for="store-port" class="text-xs text-exo-light-gray uppercase tracking-wide">Store Port</label>
              <input
                id="store-port"
                type="number"
                bind:value={storePort}
                class="w-full bg-exo-dark-gray border border-exo-medium-gray/40 rounded px-3 py-2 text-sm font-mono text-white focus:border-exo-yellow focus:outline-none"
              />
            </div>
          </div>
          <DirectoryBrowser bind:value={storePath} onselect={(p) => (storePath = p)} label="Store Path" />
        </fieldset>

        <!-- Download Policy -->
        <fieldset class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-4">
          <legend class="text-xs font-mono uppercase tracking-widest text-exo-light-gray px-2">
            Download Policy
          </legend>
          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              bind:checked={allowHfFallback}
              class="w-4 h-4 accent-exo-yellow"
            />
            <span class="text-sm">Allow HuggingFace fallback</span>
            <span class="text-xs text-exo-light-gray/50">
              Fall back to HF when model is not in store
            </span>
          </label>
        </fieldset>

        <!-- Staging -->
        <fieldset class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-4">
          <legend class="text-xs font-mono uppercase tracking-widest text-exo-light-gray px-2">
            Staging
          </legend>
          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              bind:checked={stagingEnabled}
              class="w-4 h-4 accent-exo-yellow"
            />
            <span class="text-sm">Staging enabled</span>
          </label>
          <DirectoryBrowser bind:value={nodeCachePath} onselect={(p) => (nodeCachePath = p)} label="Node Cache Path" />
          <label class="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              bind:checked={cleanupOnDeactivate}
              class="w-4 h-4 accent-exo-yellow"
            />
            <span class="text-sm">Cleanup on deactivate</span>
            <span class="text-xs text-exo-light-gray/50">
              Delete staged files when model is shut down
            </span>
          </label>
        </fieldset>

        <!-- Node Overrides -->
        <fieldset class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-4 space-y-4">
          <legend class="text-xs font-mono uppercase tracking-widest text-exo-light-gray px-2">
            Node Overrides
          </legend>
          {#each overrides as override, i}
            <div class="rounded border border-exo-medium-gray/20 bg-exo-dark-gray p-3 space-y-3">
              <div class="flex items-center gap-3">
                <input
                  type="text"
                  bind:value={override.key}
                  placeholder="hostname or node_id"
                  class="flex-1 bg-exo-black/40 border border-exo-medium-gray/40 rounded px-3 py-1.5 text-sm font-mono text-white placeholder:text-exo-light-gray/30 focus:border-exo-yellow focus:outline-none"
                />
                <button
                  type="button"
                  class="text-xs text-red-400 hover:text-red-300 font-mono uppercase"
                  onclick={() => removeOverride(i)}
                >
                  Remove
                </button>
              </div>
              <label class="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  bind:checked={override.stagingEnabled}
                  class="w-4 h-4 accent-exo-yellow"
                />
                <span class="text-sm">Staging enabled</span>
              </label>
              <DirectoryBrowser bind:value={override.nodeCachePath} onselect={(p) => (override.nodeCachePath = p)} label="Cache Path" />
              <label class="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  bind:checked={override.cleanupOnDeactivate}
                  class="w-4 h-4 accent-exo-yellow"
                />
                <span class="text-sm">Cleanup on deactivate</span>
              </label>
            </div>
          {/each}
          <button
            type="button"
            class="text-xs font-mono text-exo-yellow hover:text-exo-yellow/80 uppercase tracking-wide border border-exo-yellow/30 rounded px-3 py-1.5 hover:border-exo-yellow/60 transition-colors"
            onclick={addOverride}
          >
            + Add Override
          </button>
        </fieldset>

        <!-- Save -->
        <div class="flex items-center gap-4">
          <button
            type="submit"
            disabled={saving || !canSave}
            class="px-6 py-2 bg-exo-yellow text-exo-black font-mono text-sm uppercase tracking-wider rounded hover:bg-exo-yellow/90 disabled:opacity-50 transition-colors"
          >
            {saving ? "Saving…" : "Save"}
          </button>
          {#if enabled && !canSave}
            <div class="text-sm font-mono text-red-400">
              Store Host and Store Path are required when enabled
            </div>
          {/if}
          {#if saveMessage}
            <div class="text-sm font-mono text-exo-yellow">
              {saveMessage}
            </div>
          {/if}
          {#if saveError}
            <div class="text-sm font-mono text-red-400">
              {saveError}
            </div>
          {/if}
        </div>
      </form>
    {/if}
  </div>
</div>
