import type { RawStateResponse } from '../src/store/endpoints/cluster';

const gib = 1024 ** 3;
const nodes = ['workstation', 'gpu-server', 'compact'];
const models = ['example/Chat-32B', 'example/Reasoning-8B', 'example/Code-14B'];
const state: RawStateResponse = {
  topology: { nodes, connections: {} },
  nodeIdentities: {
    workstation: { friendlyName: 'Workstation', modelId: 'Mac mini', chipId: 'Apple M4' },
    'gpu-server': { friendlyName: 'GPU server', modelId: 'NVIDIA GPU', chipId: 'NVIDIA' },
    compact: { friendlyName: 'Compact', modelId: 'AMD Ryzen AI Max', chipId: 'AMD Ryzen AI Max' },
  },
  nodeMemory: Object.fromEntries(nodes.map((id, i) => [id, { ramTotal: { inBytes: [24, 128, 64][i] * gib }, ramAvailable: { inBytes: [11, 16, 61][i] * gib } }])),
  nodeSystem: Object.fromEntries(nodes.map((id, i) => [id, { gpuUsage: [54, 92, 5][i], ...(i === 1 ? { accelerator: { vendor: 'nvidia', name: 'Example GPU' } } : {}), temp: [30, 43, 30][i], sysPower: [10, 12, 5][i] }])),
  instances: Object.fromEntries(models.map((modelId, i) => [
    `instance-${i}`,
    { MlxRingInstance: {
      instanceId: `instance-${i}`,
      shardAssignments: {
        modelId,
        nodeToRunner: { [nodes[i]]: `runner-${i}` },
        runnerToShard: { [`runner-${i}`]: { PipelineShardMetadata: {
          modelCard: { tasks: ['TextGeneration'], placement: { compatibleBackends: ['mlx'] } },
        } } },
      },
    } },
  ])),
  runners: { 'runner-0': { RunnerReady: {} }, 'runner-1': { RunnerLoading: { layersLoaded: 12, totalLayers: 40 } }, 'runner-2': { RunnerFailed: { errorMessage: 'Fixture runner could not initialize.' } } },
  capabilityNodes: { workstation: [{ pluginId: 'example', nodeId: 'video', bundleId: 'example.video', title: 'Video Studio', version: '1.0.0', status: 'ready', ownerAvailable: true, observedAt: '2026-09-20T06:00:00Z', operationsActive: 0,
    surfaces: [{ surfaceId: 'studio', title: 'Video Studio', kind: 'link', url: '/studio', ready: true }],
    actions: [{ actionId: 'open', title: 'Open Video Studio', kind: 'open_surface', surfaceId: 'studio' }],
  }] },
  downloads: {},
};

/** Fictional API observations used only by complete-screen Storybook stories. */
export const screenFixtures: Record<string, unknown> = {
  '/state': state,
  '/node_id': 'workstation',
  '/node/identity': { nodeId: 'workstation', hostname: 'workstation.example', ipAddress: '192.0.2.10' },
  '/store/downloads': [],
  '/store/registry': { entries: models.slice(0, 2).map((model_id, i) => ({ model_id, total_bytes: [18, 5][i] * gib, files: ['model.safetensors'], downloaded_at: '2026-09-18T12:00:00Z', installed_card: { installed_identity: `fixture-card-${i}`, verification: 'local_legacy', artifact_role: 'base' } })) },
  '/models': { data: [...models, 'example/Chat-32B-8bit'].map((id, i) => ({ id, name: i === 3 ? 'Example Chat 32B' : ['Example Chat 32B', 'Example Reasoning 8B', 'Example Code 14B'][i], family: 'example', base_model: i === 3 ? models[0] : id, storage_size_megabytes: [18000, 5000, 9000, 34000][i], quantization: i === 3 ? '8-bit' : '4-bit', capabilities: ['text', 'code'], tasks: ['TextGeneration'] })) },
  '/v1/models': { object: 'list', data: models.map(id => ({ id, object: 'model', capabilities: ['text'] })) },
  '/v1/steward': { enabled: true, present: true, ready: true, steward_model: 'example/Fabric-32B', desired_model: 'example/Fabric-32B', instance_id: 'fabric-instance', transition: 'idle', progress: null, state: 'ready' },
  '/v1/connectivity/remote-access': { local: { ip: '192.0.2.10', port: 52415, url: 'http://192.0.2.10:52415' }, tailscale: { running: true, ip: '192.0.2.20', dnsName: 'cluster.example', port: 52415, url: 'http://cluster.example:52415' }, preferredUrl: 'http://cluster.example:52415', operatorUrl: null },
};
