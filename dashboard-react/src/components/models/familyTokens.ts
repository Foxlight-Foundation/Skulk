/**
 * Family tokens safe to infer from a repo name (word-boundary matched).
 * Short or ambiguous alias keys are deliberately excluded.
 */
const NAME_TOKEN_FAMILIES: readonly string[] = [
  'qwen', 'llama', 'gemma', 'deepseek', 'kimi', 'minimax', 'mistral',
  'whisper', 'flux', 'longcat', 'moonshot', 'poolside', 'laguna', 'nemotron',
  'gpt-oss', 'comfyui', 'internlm', 'baichuan', 'hunyuan', 'grok', 'phi',
];

/** Family tokens found in a repo id's name segment, in appearance order. */
export function familyTokensFromName(repoId: string): string[] {
  const tail = (repoId.includes('/') ? repoId.slice(repoId.indexOf('/') + 1) : repoId).toLowerCase();
  const found: { token: string; index: number }[] = [];
  for (const token of NAME_TOKEN_FAMILIES) {
    const index = tail.search(new RegExp(`(?:^|[^a-z0-9])${token}(?:$|[^a-z])`));
    const at = tail.startsWith(token) ? 0 : index;
    if (at !== -1) found.push({ token, index: at });
  }
  return found.sort((a, b) => a.index - b.index).map((f) => f.token);
}
