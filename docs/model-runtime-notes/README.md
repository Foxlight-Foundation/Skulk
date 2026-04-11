<!-- Copyright 2025 Foxlight Foundation -->

# Model Runtime Notes

This folder is the working corpus for model-specific runtime behavior in Skulk.

These notes are intentionally more operational and implementation-aware than the
user-facing docs in `website/docs/model-behaviors/`.

Use one Markdown file per model family or per model when the behavior is
specific enough that family-level notes would be misleading.

## What Belongs Here

Each note should capture the things we only learn by actually running the model
in Skulk, especially in clustered mode:

- prompt and parser quirks
- generator-path constraints
- warmup and prefill behavior
- KV-cache constraints
- known failure signatures
- current safe runtime choices
- validation steps

## Suggested Shape

Keep notes compact and practical. A good note usually covers:

1. what the model is
2. what is unusual about it
3. what failed in Skulk
4. what fixed it
5. what support envelope is currently trusted
6. what remains unknown

## Current Notes

- `gemma4.md`
- `nemotron.md`
