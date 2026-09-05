// Validate generated bodies with source-pinned released app schemas, without
// changing that app checkout or claiming this is physical-device evidence.
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { createRequire } = require('node:module');
const path = require('node:path');

const [repository, commit] = process.argv.slice(2);
assert(repository && path.isAbsolute(repository), 'an absolute app checkout is required');
assert(/^[a-f0-9]{40}$/.test(commit || ''), 'an exact source commit is required');
const dependencies = createRequire(path.join(repository, 'package.json'));
const typescript = dependencies('typescript');
function source(file) {
  return execFileSync('git', ['-C', repository, 'show', `${commit}:${file}`], {
    encoding: 'utf8', maxBuffer: 1024 * 1024,
  });
}
const lock = JSON.parse(source('package-lock.json'));
for (const dependency of ['typescript', 'zod']) {
  assert.equal(dependencies(`${dependency}/package.json`).version,
    lock.packages[`node_modules/${dependency}`].version, 'dependency version differs from pinned source');
}
const allowed = new Set([
  'src/transport/canonical-read-projection.ts',
  'src/domain/operator.ts', 'src/domain/speech.ts', 'src/security/base64url.ts',
]);
const cache = new Map();
function load(file) {
  assert(allowed.has(file), 'unexpected schema dependency');
  if (cache.has(file)) return cache.get(file).exports;
  const module = { exports: {} };
  cache.set(file, module);
  const compiled = typescript.transpileModule(source(file), { compilerOptions: {
    module: typescript.ModuleKind.CommonJS, target: typescript.ScriptTarget.ES2022,
  } }).outputText;
  const requireSchema = (name) => {
    if (name === 'zod') return dependencies('zod');
    assert(name.startsWith('@/'), 'unexpected external schema dependency');
    return load(`src/${name.slice(2)}.ts`);
  };
  new Function('require', 'module', 'exports', compiled)(requireSchema, module, module.exports);
  return module.exports;
}
let body = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  body += chunk;
  assert(Buffer.byteLength(body) <= 1024 * 1024, 'fixture input bound exceeded');
});
process.stdin.on('end', () => {
  const fixture = JSON.parse(body);
  const schemas = load('src/transport/canonical-read-projection.ts');
  const state = schemas.canonicalStateSchema.parse(fixture['/state']);
  assert.deepEqual(state.topology.nodes, ['synthetic-node']);
  assert(schemas.canonicalStateHasStableNodeIdentity(state));
  const snapshot = schemas.projectCanonicalReadSnapshot({
    clusterId: 'synthetic-cluster', clusterName: 'Fixture', connectionSummary: 'synthetic', state,
    models: schemas.canonicalModelListSchema.parse(fixture['/v1/models']),
    registry: schemas.canonicalStoreRegistrySchema.parse(fixture['/store/registry']),
    storage: schemas.canonicalStoreStorageSchema.parse(fixture['/store/storage']),
    downloads: schemas.canonicalStoreDownloadsSchema.parse(fixture['/store/downloads']),
  });
  assert.equal(snapshot.nodes.length, 1);
  assert.equal(snapshot.models.length, 2);
  for (const model of snapshot.models) {
    assert(model.modelId.startsWith('fixture/'));
    assert.equal(model.runtimes.length, 1);
    assert.equal(model.runtimes[0].state, 'ready');
  }
  assert.equal(snapshot.models.find((model) => model.modelId === 'fixture/generated-speech')
    .speechSynthesis.streaming, true);
  console.log(JSON.stringify({ schema: 'operator-fixture-contract.v1', source: commit,
    nodes: snapshot.nodes.length, readyModels: snapshot.models.length,
    dependencyVersions: { typescript: typescript.version, zod: dependencies('zod/package.json').version },
    physicalDeviceEvidence: false, capacityQualified: false }));
});
