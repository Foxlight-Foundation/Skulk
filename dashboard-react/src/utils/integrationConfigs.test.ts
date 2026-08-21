import { describe, expect, it } from 'vitest';

import type { InstanceCardData } from '../components/layout/InstancePanel';
import type { ModelInfo, ResolvedModelCapabilities } from '../types/models';
import {
  INTEGRATION_TOOLS,
  PLACEHOLDER_MODEL_ID,
  buildIntegrationSnippets,
  deriveDefaultTiers,
  deriveIntegrationModels,
  estimateParameterBillions,
  needsReasoningRoundTrip,
  partitionServingInstances,
  toDockerReachableUrl,
  type IntegrationModel,
  type IntegrationOptions,
} from './integrationConfigs';

/** Passthrough translator matching the shape Tolgee's `t` exposes. */
const t = ((_key: string, fallback: string) => fallback) as unknown as Parameters<
  typeof buildIntegrationSnippets
>[2];

function model(overrides: Partial<IntegrationModel> & { id: string }): IntegrationModel {
  return {
    contextLength: 0,
    supportsVision: false,
    supportsThinking: false,
    supportsThinkingToggle: false,
    supportsToolCalling: false,
    thinkingFormat: 'none',
    ...overrides,
  };
}

function options(overrides: Partial<IntegrationOptions> = {}): IntegrationOptions {
  const models = overrides.models ?? [model({ id: 'org/Big-70B', contextLength: 131072 })];
  const first = models.length > 0 ? models[0].id : PLACEHOLDER_MODEL_ID;
  return {
    apiUrl: 'http://192.168.1.50:52415',
    models,
    embeddingModels: [],
    opusModelId: first,
    sonnetModelId: first,
    haikuModelId: first,
    selectedModelId: first,
    codexFilesystemPath: '/home/operator',
    ...overrides,
  };
}

describe('estimateParameterBillions', () => {
  it('reads a parameter count out of a model id', () => {
    expect(estimateParameterBillions('org/Qwen3.8-27B-GGUF')).toBe(27);
    expect(estimateParameterBillions('org/Model-3.5B-x')).toBe(3.5);
  });

  it('returns zero when the id carries no parameter count', () => {
    expect(estimateParameterBillions('org/mystery-model')).toBe(0);
  });

  it('does not mistake a trailing word beginning with b for a size', () => {
    expect(estimateParameterBillions('org/model-4bit')).toBe(0);
  });
});

describe('deriveDefaultTiers', () => {
  it('falls back to the placeholder when nothing is ready', () => {
    const tiers = deriveDefaultTiers([]);
    expect(tiers.opusModelId).toBe(PLACEHOLDER_MODEL_ID);
    expect(tiers.haikuModelId).toBe(PLACEHOLDER_MODEL_ID);
  });

  it('maps a single model onto every tier', () => {
    const tiers = deriveDefaultTiers([model({ id: 'only/Model-8B' })]);
    expect(tiers).toEqual({
      opusModelId: 'only/Model-8B',
      sonnetModelId: 'only/Model-8B',
      haikuModelId: 'only/Model-8B',
    });
  });

  it('spreads three or more models across the tiers, largest first', () => {
    const tiers = deriveDefaultTiers([
      model({ id: 'a/Large-70B' }),
      model({ id: 'b/Middle-30B' }),
      model({ id: 'c/Small-4B' }),
    ]);
    expect(tiers).toEqual({
      opusModelId: 'a/Large-70B',
      sonnetModelId: 'b/Middle-30B',
      haikuModelId: 'c/Small-4B',
    });
  });
});

describe('needsReasoningRoundTrip', () => {
  it('is true only for thinking models that mark their reasoning', () => {
    expect(
      needsReasoningRoundTrip(
        model({ id: 'a/x', supportsThinking: true, thinkingFormat: 'token_delimited' }),
      ),
    ).toBe(true);
    expect(
      needsReasoningRoundTrip(
        model({ id: 'a/x', supportsThinking: true, thinkingFormat: 'channel_delimited' }),
      ),
    ).toBe(true);
    expect(
      needsReasoningRoundTrip(model({ id: 'a/x', supportsThinking: true, thinkingFormat: 'none' })),
    ).toBe(false);
    expect(needsReasoningRoundTrip(model({ id: 'a/x' }))).toBe(false);
  });
});

describe('toDockerReachableUrl', () => {
  it('rewrites loopback so a container can reach its host', () => {
    expect(toDockerReachableUrl('http://127.0.0.1:52415')).toBe(
      'http://host.docker.internal:52415',
    );
    expect(toDockerReachableUrl('http://localhost:52415')).toBe(
      'http://host.docker.internal:52415',
    );
  });

  it('leaves a routable address alone', () => {
    expect(toDockerReachableUrl('http://192.168.1.50:52415')).toBe('http://192.168.1.50:52415');
  });
});

describe('deriveIntegrationModels', () => {
  const instance = (modelId: string): InstanceCardData =>
    ({
      instanceId: `instance-${modelId}`,
      modelId,
      sharding: 'Pipeline',
      instanceType: 'MlxRing',
      engine: 'mlx',
      nodeStatuses: [],
      status: 'ready',
    }) as InstanceCardData;

  const capabilities = (
    overrides: Partial<ResolvedModelCapabilities>,
  ): ResolvedModelCapabilities => ({ ...overrides }) as ResolvedModelCapabilities;

  it('joins instances against catalog capability truth', () => {
    const catalog: ModelInfo[] = [
      {
        id: 'org/Vision-30B',
        context_length: 262144,
        resolved_capabilities: capabilities({
          supports_image_input: true,
          supports_thinking: true,
          thinking_format: 'token_delimited',
          supports_tool_calling: true,
        }),
      } as ModelInfo,
    ];
    const [derived] = deriveIntegrationModels([instance('org/Vision-30B')], catalog);
    expect(derived.contextLength).toBe(262144);
    expect(derived.supportsVision).toBe(true);
    expect(derived.supportsThinking).toBe(true);
    expect(derived.supportsToolCalling).toBe(true);
  });

  it('keeps a served model the catalog does not describe, with safe defaults', () => {
    const [derived] = deriveIntegrationModels([instance('org/Unlisted-7B')], []);
    expect(derived.id).toBe('org/Unlisted-7B');
    expect(derived.contextLength).toBe(0);
    expect(derived.supportsVision).toBe(false);
  });

  it('deduplicates models served by more than one instance', () => {
    const derived = deriveIntegrationModels(
      [instance('org/Same-8B'), instance('org/Same-8B')],
      [],
    );
    expect(derived).toHaveLength(1);
  });

  it('orders models largest first', () => {
    const derived = deriveIntegrationModels(
      [instance('a/Small-4B'), instance('b/Large-70B')],
      [],
    );
    expect(derived.map(entry => entry.id)).toEqual(['b/Large-70B', 'a/Small-4B']);
  });
});

describe('partitionServingInstances', () => {
  const instance = (
    modelId: string,
    overrides: Partial<InstanceCardData> = {},
  ): InstanceCardData =>
    ({
      instanceId: `instance-${modelId}`,
      modelId,
      sharding: 'Pipeline',
      instanceType: 'MlxRing',
      engine: 'mlx',
      nodeStatuses: [],
      status: 'ready',
      ...overrides,
    }) as InstanceCardData;

  const catalogEntry = (id: string, tasks: string[], tags: string[] = []): ModelInfo =>
    ({ id, tasks, tags }) as ModelInfo;

  it('keeps text-generation models on the chat side', () => {
    const { chat, embedding } = partitionServingInstances(
      [instance('org/Chat-8B')],
      [catalogEntry('org/Chat-8B', ['TextGeneration'])],
    );
    expect(chat.map(entry => entry.modelId)).toEqual(['org/Chat-8B']);
    expect(embedding).toHaveLength(0);
  });

  it('routes embedding instances away from chat recipes', () => {
    const { chat, embedding } = partitionServingInstances(
      [instance('BAAI/bge-small-en-v1.5', { isEmbedding: true })],
      [catalogEntry('BAAI/bge-small-en-v1.5', ['TextEmbedding'], ['embedding'])],
    );
    expect(chat).toHaveLength(0);
    expect(embedding.map(entry => entry.modelId)).toEqual(['BAAI/bge-small-en-v1.5']);
  });

  it('excludes speech models from chat recipes even without the embedding flag', () => {
    const { chat, embedding } = partitionServingInstances(
      [instance('org/Voice-TTS'), instance('org/Ears-STT')],
      [
        catalogEntry('org/Voice-TTS', ['TextToSpeech'], ['tts']),
        catalogEntry('org/Ears-STT', ['SpeechToText'], ['stt']),
      ],
    );
    expect(chat).toHaveLength(0);
    expect(embedding).toHaveLength(0);
  });

  it('drops instances that are not serving yet', () => {
    const { chat } = partitionServingInstances(
      [instance('org/Chat-8B', { status: 'loading' } as Partial<InstanceCardData>)],
      [catalogEntry('org/Chat-8B', ['TextGeneration'])],
    );
    expect(chat).toHaveLength(0);
  });

  it('does not let a large embedding model win the chat default', () => {
    const { chat } = partitionServingInstances(
      [
        instance('org/Embed-70B', { isEmbedding: true }),
        instance('org/Chat-4B'),
      ],
      [
        catalogEntry('org/Embed-70B', ['TextEmbedding'], ['embedding']),
        catalogEntry('org/Chat-4B', ['TextGeneration']),
      ],
    );
    const chatModels = deriveIntegrationModels(chat, []);
    expect(chatModels.map(entry => entry.id)).toEqual(['org/Chat-4B']);
  });
});

describe('buildIntegrationSnippets', () => {
  it('produces at least one snippet for every advertised tool', () => {
    for (const tool of INTEGRATION_TOOLS) {
      const snippets = buildIntegrationSnippets(tool.id, options(), t);
      expect(snippets.length).toBeGreaterThan(0);
      for (const snippet of snippets) {
        expect(snippet.body.length).toBeGreaterThan(0);
        expect(snippet.title.length).toBeGreaterThan(0);
      }
    }
  });

  it('points Claude Code at the Anthropic surface with the chosen tiers', () => {
    const [shell, settings] = buildIntegrationSnippets(
      'claude-code',
      options({
        opusModelId: 'a/Large-70B',
        sonnetModelId: 'b/Middle-30B',
        haikuModelId: 'c/Small-4B',
      }),
      t,
    );
    expect(shell.body).toContain('ANTHROPIC_BASE_URL=http://192.168.1.50:52415');
    expect(shell.body).toContain('ANTHROPIC_DEFAULT_OPUS_MODEL=a/Large-70B');
    expect(shell.body).toContain('ANTHROPIC_DEFAULT_HAIKU_MODEL=c/Small-4B');
    const parsed = JSON.parse(settings.body) as { env: Record<string, string> };
    expect(parsed.env.ANTHROPIC_DEFAULT_SONNET_MODEL).toBe('b/Middle-30B');
  });

  it('declares image input and the reasoning round-trip in the OpenCode config', () => {
    const [config] = buildIntegrationSnippets(
      'opencode',
      options({
        models: [
          model({
            id: 'org/Vision-30B',
            contextLength: 200000,
            supportsVision: true,
            supportsThinking: true,
            thinkingFormat: 'token_delimited',
          }),
          model({ id: 'org/Plain-8B', contextLength: 8192 }),
        ],
      }),
      t,
    );
    const parsed = JSON.parse(config.body) as {
      provider: { skulk: { options: { baseURL: string }; models: Record<string, Record<string, unknown>> } };
    };
    expect(parsed.provider.skulk.options.baseURL).toBe('http://192.168.1.50:52415/v1');
    expect(parsed.provider.skulk.models['org/Vision-30B'].modalities).toEqual({
      input: ['text', 'image'],
      output: ['text'],
    });
    expect(parsed.provider.skulk.models['org/Vision-30B'].interleaved).toEqual({
      field: 'reasoning_content',
    });
    expect(parsed.provider.skulk.models['org/Plain-8B'].modalities).toBeUndefined();
    expect(parsed.provider.skulk.models['org/Plain-8B'].interleaved).toBeUndefined();
    // Output is bounded even when the context window is far larger.
    expect(parsed.provider.skulk.models['org/Vision-30B'].limit).toEqual({
      context: 200000,
      output: 16384,
    });
  });

  it('ends the Hermes base URL at /v1 and carries the context length', () => {
    const [config] = buildIntegrationSnippets(
      'hermes',
      options({ models: [model({ id: 'org/Big-70B', contextLength: 131072 })] }),
      t,
    );
    expect(config.body).toContain('base_url: http://192.168.1.50:52415/v1');
    expect(config.body).toContain('provider: custom');
    expect(config.body).toContain('context_length: 131072');
  });

  it('gives AnythingLLM the generic OpenAI environment and a real token limit', () => {
    const [docker] = buildIntegrationSnippets(
      'anythingllm',
      options({ models: [model({ id: 'org/Big-70B', contextLength: 131072 })] }),
      t,
    );
    expect(docker.body).toContain("LLM_PROVIDER='generic-openai'");
    expect(docker.body).toContain(
      "GENERIC_OPEN_AI_BASE_PATH='http://192.168.1.50:52415/v1'",
    );
    expect(docker.body).toContain('GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT=131072');
  });

  it('names a serving embedding model in the AnythingLLM embedder block', () => {
    const snippets = buildIntegrationSnippets(
      'anythingllm',
      options({
        embeddingModels: [model({ id: 'BAAI/bge-small-en-v1.5', contextLength: 512 })],
      }),
      t,
    );
    const embedder = snippets.find(snippet => snippet.id === 'embedder');
    expect(embedder?.body).toContain('BAAI/bge-small-en-v1.5');
    expect(embedder?.body).toContain('512');
    // The chat block must still name the chat model, not the embedding one.
    const docker = snippets.find(snippet => snippet.id === 'docker');
    expect(docker?.body).toContain('org/Big-70B');
    expect(docker?.body).not.toContain('bge-small');
  });

  it('tells the operator to mount an embedding model when none is serving', () => {
    const snippets = buildIntegrationSnippets('anythingllm', options(), t);
    const embedder = snippets.find(snippet => snippet.id === 'embedder');
    expect(embedder?.body).toContain('mount an embedding model');
  });

  it('rewrites loopback for the Docker-hosted recipes only', () => {
    const loopback = options({ apiUrl: 'http://127.0.0.1:52415' });
    const [openWebUiDocker, openWebUiCli] = buildIntegrationSnippets(
      'open-webui',
      loopback,
      t,
    );
    expect(openWebUiDocker.body).toContain('http://host.docker.internal:52415/ollama');
    // The CLI runs on the host, so it must keep the loopback address.
    expect(openWebUiCli.body).toContain('OLLAMA_HOST=http://127.0.0.1:52415/ollama');
  });

  it('embeds the chosen filesystem path in the Codex MCP server', () => {
    const [config] = buildIntegrationSnippets(
      'codex',
      options({ codexFilesystemPath: '/srv/work' }),
      t,
    );
    expect(config.body).toContain('"/srv/work"');
    expect(config.body).toContain('base_url = "http://192.168.1.50:52415/v1"');
  });

  it('adds the image command for OpenClaw only when the model takes images', () => {
    const withVision = buildIntegrationSnippets(
      'openclaw',
      options({ models: [model({ id: 'org/Vision-30B', supportsVision: true })] }),
      t,
    )[1];
    expect(withVision.body).toContain('openclaw models set-image');

    const withoutVision = buildIntegrationSnippets(
      'openclaw',
      options({ models: [model({ id: 'org/Plain-8B' })] }),
      t,
    )[1];
    expect(withoutVision.body).not.toContain('set-image');
  });

  it('falls back to the placeholder model when nothing is ready', () => {
    const [config] = buildIntegrationSnippets(
      'opencode',
      options({ models: [], selectedModelId: PLACEHOLDER_MODEL_ID }),
      t,
    );
    expect(config.body).toContain(PLACEHOLDER_MODEL_ID);
  });
});
