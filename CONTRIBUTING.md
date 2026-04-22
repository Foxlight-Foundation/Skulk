<!-- Copyright 2025 Foxlight Foundation -->

# Contributing to Skulk

Thank you for your interest in contributing to Skulk! Skulk is maintained by [Foxlight Foundation](https://github.com/foxlight-foundation) and forked from [exo](https://github.com/exo-explore/exo).

## Getting Started

To run Skulk from source:

**Prerequisites:**
- [uv](https://github.com/astral-sh/uv) (for Python dependency management)
  ```bash
  brew install uv
  ```
- [macmon](https://github.com/vladkens/macmon) (for hardware monitoring on Apple Silicon)
  ```bash
  brew install macmon
  ```
- [node](https://github.com/nodejs/node) (for building the dashboard)
  ```bash
  brew install node
  ```
- [Nix](https://nixos.org/download/) (for `nix fmt`, `nix flake check`, and the repo dev shell)
- [rust](https://github.com/rust-lang/rustup) (to build Rust bindings, nightly for now)
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup toolchain install nightly
  ```

```bash
git clone https://github.com/foxlight-foundation/Skulk.git
cd Skulk/dashboard-react && npm install && npm run build && cd ..
uv sync
uv run skulk
```

Skulk's runtime contract on macOS follows the `uv` environment and the
official `mlx`/`mlx-metal` wheel stack. Nix is used for reproducible
development tooling and validation, not as a hidden alternate MLX runtime.

## Project Structure

Skulk is built with a mix of Rust, Python, TypeScript (React for the dashboard), and the codebase is actively evolving.

### Key directories:
- `src/exo/` — Python backend (inference, API, store, worker, routing)
- `dashboard-react/` — React dashboard (Skulk UI)
- `rust/` — Rust components (networking, libp2p, PyO3 bindings)
- `resources/inference_model_cards/` — Model metadata TOML files
- `deployment/logging/` — VictoriaLogs + Grafana stack and Vector config
- `docs/` — Technical documentation
- `docs/model-runtime-notes/` — Internal per-model clustered runtime notes

### Dashboard (React)

The Skulk dashboard is a React + TypeScript + styled-components app in `dashboard-react/`. Key areas:

- `src/components/pages/` — Top-level views (ChatView, DownloadsPage/ModelStore)
- `src/components/cluster/` — ClusterCard, PlacementManager, RunningInstanceCard
- `src/components/layout/` — HeaderNav, SettingsPanel, InstancePanel, ConversationPanel, StoreRegistryTable
- `src/components/chat/` — ChatForm, ChatMessages, ChatModelSelector
- `src/stores/` — Zustand stores (chatStore, uiStore) with localStorage/sessionStorage persistence
- `src/hooks/` — useClusterState, useConfig, useModelPicker

To run the dashboard in dev mode:
```bash
cd dashboard-react && npm run dev
```
This starts a Vite dev server on port 3000 with hot reload. The dev server proxies API calls to `http://localhost:52415` (the Skulk backend).

### Backend

- `src/exo/api/main.py` — FastAPI server (OpenAI, Claude, Ollama API compatibility)
- `src/exo/master/` — Master node (placement, election, event sourcing)
- `src/exo/worker/` — Worker node (inference, runner management, download coordination)
- `src/exo/store/` — Model store (registry, downloads, config, model optimizer)
- `src/exo/shared/` — Shared types, constants, topology
- `website/docs/` — Docusaurus documentation source, including API guide and model-capability docs

## Development Guidelines

Before starting work:

- Pull the latest source to ensure you're working with the most recent code
- Keep your changes focused — implement one feature or fix per pull request
- Avoid combining unrelated changes, even if they seem small

This makes reviews faster and helps us maintain code quality as the project evolves.

When a branch is release-worthy or bumps the project version, update both
`CHANGELOG.md` and the public docs release notes under `website/docs/` in the
same change.

## Pull Request Review Loop

When working an active PR, use this review loop:

1. Inspect the PR for new review comments, unresolved threads, and failing checks.
2. Rank each comment on the repository's 1–5 severity scale.
3. Ignore severity 1–2 comments.
4. Defer severity 3 comments unless maintainers explicitly ask for them in the current PR.
5. Fix severity 4–5 comments with the smallest correct change.
6. Add or update focused tests for every correctness fix.
7. Run focused validation before replying.
8. Reply on each addressed thread with the concrete fix.
9. Resolve only threads that are actually addressed.
10. Repeat until there are no unresolved severity 4–5 comments, or stop and escalate if the change becomes ambiguous, broad, or blocked.

## Code Style

Write pure functions where possible. Leverage the type systems available to you — Rust's type system, Python type hints, and TypeScript types. Comments should explain why you're doing something, not what the code does — especially for non-obvious decisions.

Run `nix fmt` to auto-format your code before submitting.

For the React dashboard:
- Use styled-components for styling (no CSS modules or Tailwind)
- Use Zustand for state management (not Redux or Context)
- Use individual selectors from stores to avoid unnecessary re-renders
- Follow the existing component patterns (styled components at top, component function at bottom)

## Model Cards

Skulk uses TOML-based model cards to define model metadata and capabilities. Model cards are stored in:
- `resources/inference_model_cards/` for text generation models
- `resources/image_model_cards/` for image generation models
- `~/.skulk/custom_model_cards/` for user-added custom models

### Adding a Model Card

To add a new model, create a TOML file with the following structure:

```toml
model_id = "mlx-community/Llama-3.2-1B-Instruct-4bit"
n_layers = 16
hidden_size = 2048
supports_tensor = true
tasks = ["TextGeneration"]
family = "llama"
quantization = "4bit"
base_model = "Llama 3.2 1B"
capabilities = ["text"]
context_length = 131072

[storage_size]
in_bytes = 729808896
```

### Required Fields

- `model_id`: Hugging Face model identifier
- `n_layers`: Number of transformer layers
- `hidden_size`: Hidden dimension size
- `supports_tensor`: Whether the model supports tensor parallelism
- `tasks`: List of supported tasks (`TextGeneration`, `TextToImage`, `ImageToImage`)
- `family`: Model family (e.g., "llama", "deepseek", "qwen")
- `quantization`: Quantization level (e.g., "4bit", "8bit", "bf16")
- `base_model`: Human-readable base model name
- `capabilities`: List of capabilities (e.g., `["text"]`, `["text", "thinking"]`)

### Optional Fields

- `context_length`: Maximum context window size in tokens (derived from `max_position_embeddings` in config.json)
- `components`: For multi-component models (like image models with separate text encoders and transformers)
- `uses_cfg`: Whether the model uses classifier-free guidance (for image models)
- `trust_remote_code`: Whether to allow remote code execution (defaults to `false` for security)

### Capabilities

The `capabilities` field defines what the model can do:
- `text`: Standard text generation
- `thinking`: Model supports chain-of-thought reasoning
- `image_edit`: Model supports image-to-image editing (FLUX.1-Kontext)

These coarse capability tags are intentionally broad. They help with catalog
badges and filtering, but they are not the full runtime behavior contract.

### Extended Capability Sections

Model cards can now optionally declare refined model behavior through structured
sections:

- `[reasoning]`
  - `supports_toggle`
  - `supports_budget`
  - `format`
  - `default_effort`
  - `disabled_effort`
- `[modalities]`
  - `supports_audio_input`
  - `supports_native_multimodal`
- `[tooling]`
  - `supports_tool_calling`
  - `tool_call_format`
- `[runtime]`
  - `prompt_renderer`
  - `output_parser`

These sections are optional. Existing cards still work without them.

At runtime, Skulk resolves the model card plus conservative model-family
defaults into a normalized capability profile. That resolved profile drives
model-aware reasoning defaults, prompt rendering, output parsing, and the
additive `resolved_capabilities` metadata returned by `/v1/models`.

For the full field reference and examples, see:
- [website/docs/model-cards.md](website/docs/model-cards.md)
- [website/docs/model-capabilities.md](website/docs/model-capabilities.md)

### Security Note

By default, `trust_remote_code` is set to `false` for security. Only enable it if the model explicitly requires remote code execution from the Hugging Face hub.

## Configuration

Skulk uses `skulk.yaml` for cluster configuration. Key sections:

- `model_store` — Store host, paths, staging, download settings
- `inference` — KV cache backend selection (`default`, `optiq`, `turboquant_adaptive`, etc.)
- `logging` — Centralized log aggregation (enabled toggle, ingest URL)
- `hf_token` — HuggingFace API token

Configuration can be edited directly in `skulk.yaml` or through the dashboard Settings panel. Changes made via the dashboard are synced to all nodes automatically via gossipsub.

## Centralized Logging

Skulk supports shipping structured logs from all cluster nodes to a central [VictoriaLogs](https://docs.victoriametrics.com/victorialogs/) instance via [Vector](https://vector.dev/).

### Setup

1. **Deploy the logging stack** on a central server (e.g. via Portainer):
   ```bash
   docker compose -f deployment/logging/docker-compose.yml up -d
   ```
   This starts VictoriaLogs (port 9428) and Grafana (port 3000).

2. **Configure logging** in the dashboard Settings panel, or in `skulk.yaml`:
   ```yaml
   logging:
     enabled: true
     ingest_url: http://<logging-server>:9428/insert/jsonline?_stream_fields=node_id,component&_msg_field=msg&_time_field=ts
   ```
   Settings are synced to all nodes via gossipsub.

3. **Install Vector** on each node:
   ```bash
   brew install vectordotdev/brew/vector
   ```
   (Vector is also available via `nix develop` if using the nix dev shell.)

4. **Run exo piped through Vector**:
   ```bash
   uv run skulk 2>/dev/tty | vector --config deployment/logging/vector.yaml
   ```
   stderr goes to the terminal (human-readable), stdout goes to Vector → VictoriaLogs.

5. **Update the VictoriaLogs URL** in `deployment/logging/vector.yaml` if your logging server is not at `192.168.0.118:9428`.

### Browsing Logs

- **VictoriaLogs VMUI**: `http://<logging-server>:9428/select/vmui/`
- **Grafana**: `http://<logging-server>:3000` (login with the credentials configured in your skulk.yaml logging section or set during stack deployment)

## API Adapters

Skulk supports multiple API formats through an adapter pattern. Adapters convert API-specific request formats to the internal `TextGenerationTaskParams` format and convert internal token chunks back to API-specific responses.

### Existing Adapters

- `chat_completions.py`: OpenAI Chat Completions API
- `claude.py`: Anthropic Claude Messages API
- `responses.py`: OpenAI Responses API
- `ollama.py`: Ollama API (for OpenWebUI compatibility)

For detailed API documentation, see [docs/api.md](docs/api.md).

## Testing

Skulk relies heavily on manual testing at this point in the project, but this is evolving. Before submitting a change, test both before and after to demonstrate how your change improves behavior. Do the best you can with the hardware you have available — if you need help testing, ask and we'll do our best to assist. Add automated tests where possible.

The React dashboard has Storybook stories for key components:
```bash
cd dashboard-react && npx storybook dev -p 6007
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request and follow the PR template

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub with:
- A clear description of the problem or feature
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Your environment (macOS version, hardware, etc.)

## Questions?

Open an issue or discussion on the [Skulk repository](https://github.com/foxlight-foundation/Skulk).
