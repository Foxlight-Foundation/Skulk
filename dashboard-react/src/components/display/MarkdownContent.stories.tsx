import type { Meta, StoryObj } from '@storybook/react-vite';
import { MarkdownContent } from './MarkdownContent';

const meta: Meta<typeof MarkdownContent> = {
  title: 'Display/MarkdownContent',
  component: MarkdownContent,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(700px, calc(100vw - 32px))', padding: 24, }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof MarkdownContent>;

export const CodeBlock: Story = {
  args: {
    content: `Here's some Python code:

\`\`\`python
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Print first 10 numbers
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
\`\`\`

And some inline code: \`const x = 42;\`
`,
  },
};

export const MathExpressions: Story = {
  args: {
    content: `The quadratic formula is:

$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$

Euler's identity: $e^{i\\pi} + 1 = 0$

And the Gaussian integral:

$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$
`,
  },
};

export const RichFormatting: Story = {
  args: {
    content: `# Model Architecture

## Overview

The model uses a **transformer architecture** with the following key features:

- Multi-head attention with *rotary position embeddings*
- SwiGLU activation in the feed-forward layers
- RMSNorm for layer normalization

> Note: This architecture is optimized for distributed inference across multiple Apple Silicon devices.

### Performance Table

| Metric | Value |
|--------|-------|
| Parameters | 30B |
| Context Length | 128k |
| Quantization | 4-bit |
| Memory Usage | ~18GB |

See the [documentation](https://example.com) for more details.
`,
  },
};

export const Mixed: Story = {
  args: {
    content: `## Inference Pipeline

The system distributes computation using pipeline parallelism:

\`\`\`typescript
interface ShardConfig {
  nodeId: string;
  layers: [number, number]; // [start, end]
  memoryBudget: number;     // bytes
}
\`\`\`

The total memory required is approximately $M = P \\times Q$ where $P$ is the parameter count and $Q$ is the quantization factor.

For a 70B model with 4-bit quantization:

$$M = 70 \\times 10^9 \\times 0.5 = 35\\text{GB}$$

> **Tip:** Use tensor parallelism (\`--sharding tensor\`) for Thunderbolt-connected nodes.
`,
  },
};
