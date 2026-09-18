import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ModelFilterPopover } from './ModelFilterPopover';
import { EMPTY_FILTERS, type FilterState } from '../../types/models';

const meta: Meta<typeof ModelFilterPopover> = {
  title: 'Models/ModelFilterPopover',
  component: ModelFilterPopover,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ position: 'relative', width: 'min(300px, calc(100vw - 48px))', height: 400, padding: 20 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof ModelFilterPopover>;

export const Default: Story = {
  render: () => {
    const [filters, setFilters] = useState<FilterState>({ ...EMPTY_FILTERS });
    return (
      <div style={{ position: 'relative' }}>
        <ModelFilterPopover
          filters={filters}
          onChange={setFilters}
          onClear={() => setFilters({ ...EMPTY_FILTERS })}
          onClose={() => {}}
        />
      </div>
    );
  },
};

export const WithActiveFilters: Story = {
  render: () => {
    const [filters, setFilters] = useState<FilterState>({
      capabilities: ['code', 'thinking'],
      sizeRange: { min: 10 * 1024, max: 50 * 1024 },
      downloadedOnly: true,
      readyOnly: false,
    });
    return (
      <div style={{ position: 'relative' }}>
        <ModelFilterPopover
          filters={filters}
          onChange={setFilters}
          onClear={() => setFilters({ ...EMPTY_FILTERS })}
          onClose={() => {}}
        />
      </div>
    );
  },
};
