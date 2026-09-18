import type { Meta, StoryObj } from '@storybook/react-vite';
import { Field } from './Field';

const SearchIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);

const meta: Meta<typeof Field> = {
  title: 'Common/Field',
  component: Field,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(300px, calc(100vw - 48px))', padding: 20, }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof Field>;

export const Default: Story = {
  args: { placeholder: 'Type something…', size: 'md' },
};

export const WithIcon: Story = {
  args: { placeholder: 'Search models…', size: 'md', icon: <SearchIcon /> },
};

export const Small: Story = {
  args: { placeholder: 'Small field', size: 'sm' },
};

export const Large: Story = {
  args: { placeholder: 'Large field', size: 'lg' },
};

export const Disabled: Story = {
  args: { placeholder: 'Disabled', size: 'md', disabled: true },
};

export const WithRightElement: Story = {
  args: {
    placeholder: 'Search…',
    size: 'md',
    icon: <SearchIcon />,
    rightElement: (
      <button
        style={{ all: 'unset', cursor: 'pointer', color: '#999', fontSize: 12 }}
        onClick={() => alert('clear')}
      >
        ✕
      </button>
    ),
  },
};
