import type { SkulkTranslate } from '../i18n/tolgee';
import type { InstanceCardData } from '../components/layout/InstancePanel';
import type { ModelInfo } from '../types/models';

/**
 * Connection recipes for external tools that talk to a Skulk cluster.
 *
 * Every snippet is generated from live cluster truth rather than being a static
 * example: the caller passes the models that currently have a ready instance,
 * and each builder bakes in real model ids, real context windows, and the
 * per-model flags a given tool needs (image input, reasoning round-trip, tool
 * calling). A cluster with nothing mounted still renders usable scaffolding
 * through {@link PLACEHOLDER_MODEL_ID}.
 *
 * Skulk exposes three request surfaces and each recipe picks whichever one its
 * tool speaks natively: OpenAI-compatible at `<api>/v1`, Anthropic-compatible
 * at `<api>/v1/messages`, and Ollama-compatible at `<api>/ollama`.
 */

/** Identifier for a supported external tool. */
export type IntegrationToolId =
  | 'claude-code'
  | 'opencode'
  | 'codex'
  | 'hermes'
  | 'openclaw'
  | 'pi'
  | 'anythingllm'
  | 'open-webui'
  | 'n8n'
  | 'firefox';

/** Which Skulk request surface a tool connects through. */
export type IntegrationSurface = 'openai' | 'anthropic' | 'ollama' | 'dashboard';

/** Syntax hint for rendering a snippet body. */
export type SnippetLanguage = 'bash' | 'json' | 'toml' | 'yaml' | 'text';

/**
 * A ready model, reduced to the facts the recipes depend on.
 *
 * Built by joining the models that currently have a ready instance against the
 * capability truth in `/models`, so the page never advertises a model the
 * cluster cannot actually serve.
 */
export interface IntegrationModel {
  /** Fully qualified Skulk model id, used verbatim as the wire model name. */
  readonly id: string;
  /** Maximum context window in tokens; `0` when the cluster reported none. */
  readonly contextLength: number;
  /** Whether the model accepts image input. */
  readonly supportsVision: boolean;
  /** Whether the model emits reasoning or thinking content. */
  readonly supportsThinking: boolean;
  /** Whether the caller can turn thinking on and off per request. */
  readonly supportsThinkingToggle: boolean;
  /** Whether the model supports tool calling on the default request path. */
  readonly supportsToolCalling: boolean;
  /** Reasoning marker format resolved from the model card. */
  readonly thinkingFormat: string;
}

/** One copy-paste block shown on the integrations page. */
export interface IntegrationSnippet {
  /** Stable key, unique within a tool. */
  readonly id: string;
  /** Translated heading, for example "Config file". */
  readonly title: string;
  /**
   * Where the content belongs, for example `~/.codex/config.toml`.
   *
   * Literal paths and URLs are left untranslated on purpose; prose subtitles
   * are translated by the builder before they land here.
   */
  readonly subtitle: string;
  /** Translated sentence explaining what the block does. */
  readonly description: string;
  /** Syntax hint for the code block. */
  readonly language: SnippetLanguage;
  /** The literal text the operator copies. */
  readonly body: string;
}

/** Everything a builder needs to render a tool's snippets. */
export interface IntegrationOptions {
  /** Cluster API origin, reachable from other machines, no trailing slash. */
  readonly apiUrl: string;
  /** Models with a ready instance, largest first. */
  readonly models: readonly IntegrationModel[];
  /** Model mapped onto Claude Code's Opus tier. */
  readonly opusModelId: string;
  /** Model mapped onto Claude Code's Sonnet tier. */
  readonly sonnetModelId: string;
  /** Model mapped onto Claude Code's Haiku tier. */
  readonly haikuModelId: string;
  /** Model chosen for single-model tools. */
  readonly selectedModelId: string;
  /** Directory exposed to Codex through the filesystem MCP server. */
  readonly codexFilesystemPath: string;
}

/** Stand-in model id used when the cluster has nothing ready. */
export const PLACEHOLDER_MODEL_ID = 'your-model-id';

/**
 * Placeholder API key.
 *
 * Skulk does not authenticate requests on the trusted fabric, but most clients
 * refuse to start without a non-empty key, so the recipes ship a dummy value.
 */
export const PLACEHOLDER_API_KEY = 'skulk';

/** Request timeout handed to clients, in milliseconds. */
const LONG_REQUEST_TIMEOUT_MS = 3_000_000;

/** Upper bound on the output budget advertised to tools that want one. */
const MAX_ADVERTISED_OUTPUT_TOKENS = 16_384;

/** Fallback token limit for tools that insist on a number. */
const FALLBACK_TOKEN_LIMIT = 4096;

/** Display metadata for one tool. */
export interface IntegrationToolDescriptor {
  readonly id: IntegrationToolId;
  /** Product name, spelled as its own documentation spells it. Not translated. */
  readonly label: string;
  /** Which Skulk surface the recipe uses. */
  readonly surface: IntegrationSurface;
  /** Whether the recipe renders a single-model chooser. */
  readonly usesSingleModelChooser: boolean;
  /** Whether the recipe renders the Opus, Sonnet and Haiku tier chooser. */
  readonly usesTierChooser: boolean;
  /** Whether the recipe renders the Codex filesystem-path input. */
  readonly usesFilesystemPath: boolean;
}

/**
 * Every supported tool, in the order the chooser lists them.
 *
 * Coding agents come first because they are why most operators open this page,
 * then the chat and workflow applications.
 */
export const INTEGRATION_TOOLS: readonly IntegrationToolDescriptor[] = [
  {
    id: 'claude-code',
    label: 'Claude Code',
    surface: 'anthropic',
    usesSingleModelChooser: false,
    usesTierChooser: true,
    usesFilesystemPath: false,
  },
  {
    id: 'opencode',
    label: 'OpenCode',
    surface: 'openai',
    usesSingleModelChooser: false,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
  {
    id: 'codex',
    label: 'Codex',
    surface: 'openai',
    usesSingleModelChooser: true,
    usesTierChooser: false,
    usesFilesystemPath: true,
  },
  {
    id: 'hermes',
    label: 'Hermes',
    surface: 'openai',
    usesSingleModelChooser: true,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
  {
    id: 'openclaw',
    label: 'OpenClaw',
    surface: 'openai',
    usesSingleModelChooser: true,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
  {
    id: 'pi',
    label: 'Pi',
    surface: 'openai',
    usesSingleModelChooser: true,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
  {
    id: 'anythingllm',
    label: 'AnythingLLM',
    surface: 'openai',
    usesSingleModelChooser: true,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
  {
    id: 'open-webui',
    label: 'Open WebUI',
    surface: 'ollama',
    usesSingleModelChooser: true,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
  {
    id: 'n8n',
    label: 'n8n',
    surface: 'openai',
    usesSingleModelChooser: true,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
  {
    id: 'firefox',
    label: 'Firefox',
    surface: 'dashboard',
    usesSingleModelChooser: false,
    usesTierChooser: false,
    usesFilesystemPath: false,
  },
];

/**
 * Rewrites a loopback origin so a container can reach its host.
 *
 * Docker-hosted tools resolve `localhost` to their own container, so recipes
 * that run in Docker must dial `host.docker.internal` instead.
 */
export function toDockerReachableUrl(apiUrl: string): string {
  return apiUrl
    .replace('127.0.0.1', 'host.docker.internal')
    .replace('localhost', 'host.docker.internal');
}

/**
 * Reads a parameter count in billions out of a model id.
 *
 * Used only to order models for the default Claude Code tier mapping; an
 * unreadable id sorts last rather than failing.
 */
export function estimateParameterBillions(modelId: string): number {
  const match = /(\d+(?:\.\d+)?)\s*[Bb](?![a-zA-Z])/.exec(modelId);
  return match ? Number.parseFloat(match[1]) : 0;
}

/** Sorts ready models largest first, which every default below assumes. */
export function sortModelsBySize(models: readonly IntegrationModel[]): IntegrationModel[] {
  return [...models].sort(
    (a, b) => estimateParameterBillions(b.id) - estimateParameterBillions(a.id),
  );
}

/**
 * Joins ready instances against the model catalog.
 *
 * An instance whose model is missing from the catalog still appears, with
 * conservative capability defaults, because the cluster can plainly serve it.
 */
export function deriveIntegrationModels(
  readyInstances: readonly InstanceCardData[],
  catalog: readonly ModelInfo[],
): IntegrationModel[] {
  const byId = new Map(catalog.map((model) => [model.id, model]));
  const seen = new Set<string>();
  const derived: IntegrationModel[] = [];
  for (const instance of readyInstances) {
    if (!instance.modelId || seen.has(instance.modelId)) continue;
    seen.add(instance.modelId);
    const info = byId.get(instance.modelId);
    const resolved = info?.resolved_capabilities;
    derived.push({
      id: instance.modelId,
      contextLength: info?.context_length ?? 0,
      supportsVision: resolved?.supports_image_input ?? false,
      supportsThinking: resolved?.supports_thinking ?? false,
      supportsThinkingToggle: resolved?.supports_thinking_toggle ?? false,
      supportsToolCalling: resolved?.supports_tool_calling ?? false,
      thinkingFormat: resolved?.thinking_format ?? 'none',
    });
  }
  return sortModelsBySize(derived);
}

/**
 * Picks default Opus, Sonnet and Haiku models for Claude Code.
 *
 * The largest model answers as Opus, the middle of the range as Sonnet and the
 * smallest as Haiku, so the tiers keep their usual capability shape. The
 * operator may override any tier.
 */
export function deriveDefaultTiers(models: readonly IntegrationModel[]): {
  opusModelId: string;
  sonnetModelId: string;
  haikuModelId: string;
} {
  const count = models.length;
  if (count === 0) {
    return {
      opusModelId: PLACEHOLDER_MODEL_ID,
      sonnetModelId: PLACEHOLDER_MODEL_ID,
      haikuModelId: PLACEHOLDER_MODEL_ID,
    };
  }
  const largest = models[0].id;
  const smallest = models[count - 1].id;
  if (count === 1) {
    return { opusModelId: largest, sonnetModelId: largest, haikuModelId: largest };
  }
  if (count === 2) {
    return { opusModelId: largest, sonnetModelId: smallest, haikuModelId: smallest };
  }
  return {
    opusModelId: largest,
    sonnetModelId: models[Math.floor(count / 2)].id,
    haikuModelId: smallest,
  };
}

/**
 * Whether a model's reasoning should be fed back on later turns.
 *
 * Families that mark reasoning with delimiters read the previous turn's
 * reasoning back out of the transcript, so clients that support it should
 * return `reasoning_content` rather than dropping it. Models without reasoning
 * markers gain nothing from the round-trip.
 */
export function needsReasoningRoundTrip(model: IntegrationModel): boolean {
  return model.supportsThinking && model.thinkingFormat !== 'none';
}

/** Resolves a model id back to its record. */
function findModel(options: IntegrationOptions, modelId: string): IntegrationModel | undefined {
  return options.models.find((model) => model.id === modelId);
}

/** First ready model id, or the placeholder when nothing is ready. */
function primaryModelId(options: IntegrationOptions): string {
  return options.models.length > 0 ? options.models[0].id : PLACEHOLDER_MODEL_ID;
}

/** Token limit for a model, falling back when the cluster reported none. */
function tokenLimitFor(options: IntegrationOptions, modelId: string): number {
  const model = findModel(options, modelId);
  return model && model.contextLength > 0 ? model.contextLength : FALLBACK_TOKEN_LIMIT;
}

function buildClaudeCodeSnippets(
  options: IntegrationOptions,
  t: SkulkTranslate,
): IntegrationSnippet[] {
  const environment: Record<string, string> = {
    ANTHROPIC_BASE_URL: options.apiUrl,
    ANTHROPIC_API_KEY: PLACEHOLDER_API_KEY,
    ANTHROPIC_DEFAULT_OPUS_MODEL: options.opusModelId,
    ANTHROPIC_DEFAULT_SONNET_MODEL: options.sonnetModelId,
    ANTHROPIC_DEFAULT_HAIKU_MODEL: options.haikuModelId,
    API_TIMEOUT_MS: String(LONG_REQUEST_TIMEOUT_MS),
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: '1',
  };
  return [
    {
      id: 'shell',
      title: t('integrations.snippet.shellCommand', 'Shell command'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.claudeCode.shell.description',
        'Starts Claude Code against the cluster for this shell only. The long timeout covers large local models that take a while to reach a first token.',
      ),
      language: 'bash',
      body: [
        ...Object.entries(environment).map(([key, value]) => `${key}=${value} \\`),
        'claude',
      ].join('\n'),
    },
    {
      id: 'settings',
      title: t('integrations.snippet.settingsFile', 'Settings file'),
      subtitle: '~/.claude/settings.json',
      description: t(
        'integrations.claudeCode.settings.description',
        'Persists the same configuration so every Claude Code session uses the cluster.',
      ),
      language: 'json',
      body: JSON.stringify({ env: environment }, null, 2),
    },
  ];
}

function buildOpenCodeSnippets(
  options: IntegrationOptions,
  t: SkulkTranslate,
): IntegrationSnippet[] {
  const models: Record<string, Record<string, unknown>> = {};
  for (const model of options.models) {
    const entry: Record<string, unknown> = { name: model.id };
    if (model.contextLength > 0) {
      entry.limit = {
        context: model.contextLength,
        output: Math.min(model.contextLength, MAX_ADVERTISED_OUTPUT_TOKENS),
      };
    }
    if (model.supportsVision) {
      entry.modalities = { input: ['text', 'image'], output: ['text'] };
    }
    if (needsReasoningRoundTrip(model)) {
      entry.interleaved = { field: 'reasoning_content' };
    }
    models[model.id] = entry;
  }
  if (Object.keys(models).length === 0) {
    models[PLACEHOLDER_MODEL_ID] = { name: PLACEHOLDER_MODEL_ID };
  }
  return [
    {
      id: 'config',
      title: t('integrations.snippet.configFile', 'Config file'),
      subtitle: 'opencode.json',
      description: t(
        'integrations.opencode.config.description',
        'Put this in your project root, or in ~/.config/opencode/opencode.json to use the cluster everywhere. Vision models declare image input, and reasoning models are set to send their prior reasoning back on later turns.',
      ),
      language: 'json',
      body: JSON.stringify(
        {
          $schema: 'https://opencode.ai/config.json',
          provider: {
            skulk: {
              npm: '@ai-sdk/openai-compatible',
              name: 'Skulk',
              options: { baseURL: `${options.apiUrl}/v1`, apiKey: PLACEHOLDER_API_KEY },
              models,
            },
          },
          model: `skulk/${primaryModelId(options)}`,
        },
        null,
        2,
      ),
    },
  ];
}

function buildCodexSnippets(options: IntegrationOptions, t: SkulkTranslate): IntegrationSnippet[] {
  return [
    {
      id: 'config',
      title: t('integrations.snippet.configFile', 'Config file'),
      subtitle: '~/.codex/config.toml',
      description: t(
        'integrations.codex.config.description',
        'Registers the cluster as a Codex model provider and gives Codex filesystem access to the directory above.',
      ),
      language: 'toml',
      body: [
        `model = "${options.selectedModelId}"`,
        'model_provider = "skulk"',
        '',
        '[model_providers.skulk]',
        'name = "Skulk"',
        `base_url = "${options.apiUrl}/v1"`,
        'env_key = "SKULK_API_KEY"',
        '',
        '[mcp_servers.filesystem]',
        'command = "npx"',
        `args = ["-y", "@modelcontextprotocol/server-filesystem", "${options.codexFilesystemPath}"]`,
      ].join('\n'),
    },
    {
      id: 'shell',
      title: t('integrations.snippet.shellCommand', 'Shell command'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.codex.shell.description',
        'Starts Codex with the provider above. The key is unused but must be present.',
      ),
      language: 'bash',
      body: `SKULK_API_KEY=${PLACEHOLDER_API_KEY} npx @openai/codex`,
    },
  ];
}

function buildHermesSnippets(options: IntegrationOptions, t: SkulkTranslate): IntegrationSnippet[] {
  const model = findModel(options, options.selectedModelId);
  const contextLine =
    model && model.contextLength > 0 ? `  context_length: ${model.contextLength}\n` : '';
  return [
    {
      id: 'config',
      title: t('integrations.snippet.configFile', 'Config file'),
      subtitle: '~/.hermes/config.yaml',
      description: t(
        'integrations.hermes.config.description',
        'Points Hermes Agent at the cluster through its custom endpoint provider. The base URL ends at /v1 because Hermes appends the chat-completions path itself.',
      ),
      language: 'yaml',
      body:
        'model:\n' +
        `  default: ${options.selectedModelId}\n` +
        '  provider: custom\n' +
        `  base_url: ${options.apiUrl}/v1\n` +
        `  api_key: ${PLACEHOLDER_API_KEY}\n` +
        contextLine,
    },
    {
      id: 'interactive',
      title: t('integrations.snippet.interactiveSetup', 'Interactive setup'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.hermes.interactive.description',
        'Or configure it inside Hermes: choose the custom endpoint option, then give it the same base URL, key and model.',
      ),
      language: 'bash',
      body: 'hermes model',
    },
    {
      id: 'timeout',
      title: t('integrations.snippet.streamTimeout', 'Stream timeout'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.hermes.timeout.description',
        'Large models on a local cluster can out-wait the default stream read timeout during long agentic turns. Raise it before launching Hermes.',
      ),
      language: 'bash',
      body: 'HERMES_STREAM_READ_TIMEOUT=600 hermes',
    },
  ];
}

function buildOpenClawSnippets(
  options: IntegrationOptions,
  t: SkulkTranslate,
): IntegrationSnippet[] {
  const model = findModel(options, options.selectedModelId);
  return [
    {
      id: 'config',
      title: t('integrations.snippet.configFile', 'Config file'),
      subtitle: '~/.openclaw/openclaw.json',
      description: t(
        'integrations.openclaw.config.description',
        'Registers the cluster as an OpenClaw provider. Install OpenClaw first with npm install -g openclaw@latest',
      ),
      language: 'json',
      body: JSON.stringify(
        {
          gateway: { mode: 'local' },
          models: {
            providers: {
              skulk: {
                baseUrl: `${options.apiUrl}/v1`,
                apiKey: PLACEHOLDER_API_KEY,
                api: 'openai-completions',
                models: [
                  {
                    id: options.selectedModelId,
                    name: 'Skulk cluster',
                    input: model?.supportsVision ? ['text', 'image'] : ['text'],
                  },
                ],
              },
            },
          },
          agents: { defaults: { model: `skulk/${options.selectedModelId}` } },
        },
        null,
        2,
      ),
    },
    {
      id: 'setup',
      title: t('integrations.snippet.setupCommands', 'Setup commands'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.openclaw.setup.description',
        'Reconciles provider metadata, then starts the gateway and dashboard.',
      ),
      language: 'bash',
      body: [
        'openclaw doctor --fix',
        ...(model?.supportsVision
          ? [`openclaw models set-image skulk/${options.selectedModelId}`]
          : []),
        'openclaw gateway &',
        'openclaw dashboard',
      ].join('\n'),
    },
  ];
}

function buildPiSnippets(options: IntegrationOptions, t: SkulkTranslate): IntegrationSnippet[] {
  const models = options.models.map((model) => {
    const entry: Record<string, unknown> = { id: model.id };
    if (model.supportsVision) entry.input = ['text', 'image'];
    if (model.supportsThinking || model.supportsThinkingToggle) entry.reasoning = true;
    if (model.contextLength > 0) entry.contextWindow = model.contextLength;
    return entry;
  });
  if (models.length === 0) models.push({ id: PLACEHOLDER_MODEL_ID });
  return [
    {
      id: 'models',
      title: t('integrations.snippet.modelsConfig', 'Models config'),
      subtitle: '~/.pi/agent/models.json',
      description: t(
        'integrations.pi.models.description',
        'Registers the cluster as a Pi provider, then pick a model with /model. Install Pi with npm install -g @mariozechner/pi-coding-agent',
      ),
      language: 'json',
      body: JSON.stringify(
        {
          providers: {
            skulk: {
              baseUrl: `${options.apiUrl}/v1`,
              api: 'openai-completions',
              apiKey: PLACEHOLDER_API_KEY,
              compat: {
                supportsDeveloperRole: false,
                supportsReasoningEffort: false,
                thinkingFormat: 'qwen',
              },
              models,
            },
          },
        },
        null,
        2,
      ),
    },
    {
      id: 'shell',
      title: t('integrations.snippet.shellCommand', 'Shell command'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.pi.shell.description',
        'Starts Pi with the cluster provider and model already selected.',
      ),
      language: 'bash',
      body: `pi --provider skulk --model ${options.selectedModelId}`,
    },
  ];
}

function buildAnythingLlmSnippets(
  options: IntegrationOptions,
  t: SkulkTranslate,
): IntegrationSnippet[] {
  const dockerUrl = toDockerReachableUrl(options.apiUrl);
  const tokenLimit = tokenLimitFor(options, options.selectedModelId);
  return [
    {
      id: 'docker',
      title: t('integrations.anythingllm.docker.title', 'Start AnythingLLM'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.anythingllm.docker.description',
        'Starts AnythingLLM already pointed at the cluster. Skip this if you run the desktop app and use the settings below instead.',
      ),
      language: 'bash',
      body: [
        'docker run -d -p 3001:3001 \\',
        "  -e LLM_PROVIDER='generic-openai' \\",
        `  -e GENERIC_OPEN_AI_BASE_PATH='${dockerUrl}/v1' \\`,
        `  -e GENERIC_OPEN_AI_MODEL_PREF='${options.selectedModelId}' \\`,
        `  -e GENERIC_OPEN_AI_MODEL_TOKEN_LIMIT=${tokenLimit} \\`,
        `  -e GENERIC_OPEN_AI_API_KEY='${PLACEHOLDER_API_KEY}' \\`,
        '  -v anythingllm_storage:/app/server/storage \\',
        '  --name anythingllm \\',
        '  mintplexlabs/anythingllm',
      ].join('\n'),
    },
    {
      id: 'ui',
      title: t('integrations.anythingllm.ui.title', 'Or configure it in the app'),
      subtitle: 'Settings, AI Providers, LLM',
      description: t(
        'integrations.anythingllm.ui.description',
        'The desktop app takes the same values through its settings screen. Generic OpenAI is the provider that accepts a custom base URL.',
      ),
      language: 'text',
      body: [
        `Provider: Generic OpenAI`,
        `Base URL: ${options.apiUrl}/v1`,
        `API Key: ${PLACEHOLDER_API_KEY}`,
        `Chat Model Name: ${options.selectedModelId}`,
        `Token Context Window: ${tokenLimit}`,
      ].join('\n'),
    },
    {
      id: 'embedder',
      title: t('integrations.anythingllm.embedder.title', 'Embed on the cluster too'),
      subtitle: 'Settings, AI Providers, Embedder',
      description: t(
        'integrations.anythingllm.embedder.description',
        'AnythingLLM indexes documents with an embedding model. Skulk serves embeddings on the same surface, so the cluster can do that work as well. Mount an embedding model and use its id here.',
      ),
      language: 'text',
      body: [
        `Embedding Engine: Generic OpenAI`,
        `Base URL: ${options.apiUrl}/v1`,
        `API Key: ${PLACEHOLDER_API_KEY}`,
      ].join('\n'),
    },
  ];
}

function buildOpenWebUiSnippets(
  options: IntegrationOptions,
  t: SkulkTranslate,
): IntegrationSnippet[] {
  const dockerUrl = toDockerReachableUrl(options.apiUrl);
  return [
    {
      id: 'docker',
      title: t('integrations.openWebui.docker.title', 'Start Open WebUI'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.openWebui.docker.description',
        "Starts Open WebUI against the cluster's Ollama-compatible surface.",
      ),
      language: 'bash',
      body: [
        'docker run -d -p 3000:8080 \\',
        `  -e OLLAMA_BASE_URL=${dockerUrl}/ollama \\`,
        '  -v open-webui:/app/backend/data \\',
        '  --name open-webui \\',
        '  ghcr.io/open-webui/open-webui:main',
      ].join('\n'),
    },
    {
      id: 'cli',
      title: t('integrations.openWebui.cli.title', 'Ollama CLI'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t(
        'integrations.openWebui.cli.description',
        'The same surface works with the stock Ollama client, with no Skulk-specific flags.',
      ),
      language: 'bash',
      body: `OLLAMA_HOST=${options.apiUrl}/ollama ollama run ${primaryModelId(options)}`,
    },
  ];
}

function buildN8nSnippets(options: IntegrationOptions, t: SkulkTranslate): IntegrationSnippet[] {
  const dockerUrl = toDockerReachableUrl(options.apiUrl);
  return [
    {
      id: 'docker',
      title: t('integrations.n8n.docker.title', 'Start n8n'),
      subtitle: t('integrations.snippet.runInTerminal', 'Run in a terminal'),
      description: t('integrations.n8n.docker.description', 'Skip this if you already run n8n.'),
      language: 'bash',
      body: [
        'docker run -d -p 5678:5678 \\',
        '  -v n8n_data:/home/node/.n8n \\',
        '  --name n8n \\',
        '  docker.n8n.io/n8nio/n8n',
      ].join('\n'),
    },
    {
      id: 'credential',
      title: t('integrations.n8n.credential.title', 'Add an OpenAI credential'),
      subtitle: 'n8n, Credentials',
      description: t(
        'integrations.n8n.credential.description',
        "Point n8n's stock OpenAI credential at the cluster instead of OpenAI.",
      ),
      language: 'text',
      body: [
        `Credential type: OpenAI API`,
        `API Key: ${PLACEHOLDER_API_KEY}`,
        `Base URL: ${dockerUrl}/v1`,
      ].join('\n'),
    },
    {
      id: 'workflow',
      title: t('integrations.n8n.workflow.title', 'Build a workflow'),
      subtitle: 'n8n, Workflows',
      description: t(
        'integrations.n8n.workflow.description',
        "Skulk model ids are not in n8n's built-in model list, so enter the id directly rather than picking from the dropdown.",
      ),
      language: 'text',
      body: [
        `1. Add an "AI Agent" or "Basic LLM Chain" node`,
        `2. Inside it, add an "OpenAI Chat Model" sub-node`,
        `3. Select the credential you saved`,
        `4. Set Model to "By ID" and enter ${primaryModelId(options)}`,
        `5. Connect a "Chat Trigger" node to talk to it`,
      ].join('\n'),
    },
  ];
}

function buildFirefoxSnippets(
  options: IntegrationOptions,
  t: SkulkTranslate,
): IntegrationSnippet[] {
  return [
    {
      id: 'about-config',
      title: t('integrations.firefox.title', 'Firefox AI chatbot'),
      subtitle: 'about:config',
      description: t(
        'integrations.firefox.description',
        "Uses this dashboard's chat as Firefox's built-in AI sidebar. Requires Firefox 130 or newer.",
      ),
      language: 'text',
      body: [
        `1. Open about:config`,
        `2. Set browser.ml.chat.enabled to true`,
        `3. Set browser.ml.chat.hideLocalhost to false`,
        `4. Set browser.ml.chat.provider to ${options.apiUrl}/`,
      ].join('\n'),
    },
  ];
}

const SNIPPET_BUILDERS: Record<
  IntegrationToolId,
  (options: IntegrationOptions, t: SkulkTranslate) => IntegrationSnippet[]
> = {
  'claude-code': buildClaudeCodeSnippets,
  opencode: buildOpenCodeSnippets,
  codex: buildCodexSnippets,
  hermes: buildHermesSnippets,
  openclaw: buildOpenClawSnippets,
  pi: buildPiSnippets,
  anythingllm: buildAnythingLlmSnippets,
  'open-webui': buildOpenWebUiSnippets,
  n8n: buildN8nSnippets,
  firefox: buildFirefoxSnippets,
};

/**
 * Builds the copy-paste blocks for one tool.
 *
 * Pure with respect to cluster state: the same options and translator always
 * produce the same snippets, which is what makes this testable without a
 * browser.
 */
export function buildIntegrationSnippets(
  toolId: IntegrationToolId,
  options: IntegrationOptions,
  t: SkulkTranslate,
): IntegrationSnippet[] {
  return SNIPPET_BUILDERS[toolId](options, t);
}
