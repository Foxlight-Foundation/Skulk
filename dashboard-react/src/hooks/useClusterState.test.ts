import { describe, expect, it } from 'vitest';

import { transformTopology } from './useClusterState';

const GIB = 1024 ** 3;

/**
 * Memory-pool selection for the node card (#669): discrete-GPU nodes show
 * their VRAM pool (the pool models serve from), Strix-style carve-out nodes
 * show the reassembled unified pool, and unified-memory nodes show system
 * RAM unchanged.
 */
describe('transformTopology memory pool selection', () => {
  function transformOne(
    memory: Parameters<typeof transformTopology>[2][string],
    system: Parameters<typeof transformTopology>[3][string],
  ) {
    const topology = transformTopology(
      { nodes: ['node-a'] },
      {},
      { 'node-a': memory },
      { 'node-a': system },
      {},
      {},
      {},
      {},
    );
    return topology.nodes['node-a'];
  }

  it('shows the VRAM pool for a discrete NVIDIA node', () => {
    // The #669 shape: a GPU-cloud host with far more system RAM than VRAM.
    const node = transformOne(
      {
        ramTotal: { inBytes: 503 * GIB },
        ramAvailable: { inBytes: 400 * GIB },
      },
      {
        accelerator: {
          vendor: 'nvidia',
          name: 'NVIDIA A40',
          vramTotalBytes: 45 * GIB,
          vramUsedBytes: 30 * GIB,
        },
      },
    );
    expect(node.mactop_info?.memory).toEqual({
      ram_usage: 30 * GIB,
      ram_total: 45 * GIB,
      is_vram: true,
    });
    expect(node.system_info.memory).toBe(45 * GIB);
  });

  it('reassembles the unified pool for an AMD carve-out node', () => {
    // Strix Halo: the BIOS VRAM carve-out is not counted as system RAM, so
    // the full unified pool is their sum.
    const node = transformOne(
      {
        ramTotal: { inBytes: 96 * GIB },
        ramAvailable: { inBytes: 90 * GIB },
      },
      {
        accelerator: {
          vendor: 'amd',
          name: 'AMD Radeon Graphics',
          vramTotalBytes: 32 * GIB,
          vramUsedBytes: 2 * GIB,
        },
      },
    );
    expect(node.mactop_info?.memory).toEqual({
      ram_usage: 6 * GIB + 2 * GIB,
      ram_total: 128 * GIB,
      is_vram: undefined,
    });
  });

  it('keeps plain system RAM for a unified-memory node without reported VRAM', () => {
    const node = transformOne(
      {
        ramTotal: { inBytes: 16 * GIB },
        ramAvailable: { inBytes: 8 * GIB },
      },
      { accelerator: { vendor: 'apple', name: 'Apple GPU' } },
    );
    expect(node.mactop_info?.memory).toEqual({
      ram_usage: 8 * GIB,
      ram_total: 16 * GIB,
      is_vram: undefined,
    });
  });
});
