import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { SearchBar } from './SearchBar';

const meta: Meta<typeof SearchBar> = {
  title: 'Common/SearchBar',
  component: SearchBar,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(320px, calc(100vw - 48px))', padding: 20, }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof SearchBar>;

export const Default: Story = {
  render: () => {
    const [value, setValue] = useState('');
    return <SearchBar value={value} onChange={setValue} placeholder="Search models…" />;
  },
};

export const WithValue: Story = {
  render: () => {
    const [value, setValue] = useState('llama 70b');
    return <SearchBar value={value} onChange={setValue} placeholder="Search models…" />;
  },
};

export const Debounced: Story = {
  render: () => {
    const [value, setValue] = useState('');
    const [committed, setCommitted] = useState('');
    return (
      <div>
        <SearchBar
          value={value}
          onChange={(v) => {
            setValue(v);
            setCommitted(v);
          }}
          placeholder="Debounced (500ms)…"
          debounceMs={500}
        />
        <p style={{ color: 'inherit', fontSize: 12, marginTop: 8 }}>
          Committed: "{committed}"
        </p>
      </div>
    );
  },
};

export const Small: Story = {
  render: () => {
    const [value, setValue] = useState('');
    return <SearchBar value={value} onChange={setValue} size="sm" placeholder="Small search…" />;
  },
};
