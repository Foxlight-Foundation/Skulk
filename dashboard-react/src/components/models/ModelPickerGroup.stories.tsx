import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ModelPickerGroup } from './ModelPickerGroup';
import type { ModelGroup, ModelFitStatus } from '../../types/models';

const GB = 1024; // MB

function makeGroup(overrides: Partial<ModelGroup> = {}): ModelGroup {
  return {
    id: 'qwen3-30b',
    name: 'Qwen3 30B A3B',
    capabilities: ['text', 'thinking', 'code'],
    family: 'qwen',
    variants: [
      { id: 'qwen3-30b-4bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 18 * GB, quantization: '4-bit', base_model: 'qwen3-30b', capabilities: ['text', 'thinking', 'code'], family: 'qwen' },
      { id: 'qwen3-30b-8bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 32 * GB, quantization: '8-bit', base_model: 'qwen3-30b', capabilities: ['text', 'thinking', 'code'], family: 'qwen' },
    ],
    smallestVariant: { id: 'qwen3-30b-4bit', name: 'Qwen3 30B A3B', storage_size_megabytes: 18 * GB, quantization: '4-bit', base_model: 'qwen3-30b' },
    hasMultipleVariants: true,
    ...overrides,
  };
}

function makeSingleGroup(): ModelGroup {
  const v = { id: 'llama-8b-4bit', name: 'Llama 3.1 8B', storage_size_megabytes: 5 * GB, quantization: '4-bit', base_model: 'llama-8b', capabilities: ['text', 'code'], family: 'llama' };
  return {
    id: 'llama-8b',
    name: 'Llama 3.1 8B',
    capabilities: ['text', 'code'],
    family: 'llama',
    variants: [v],
    smallestVariant: v,
    hasMultipleVariants: false,
  };
}

const fitAll = () => true;
const fitStatus = (): ModelFitStatus => 'fits_now';

const meta: Meta<typeof ModelPickerGroup> = {
  title: 'Models/ModelPickerGroup',
  component: ModelPickerGroup,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(760px, calc(100vw - 32px))', padding: 8, borderRadius: 8 }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof ModelPickerGroup>;

export const MultiVariant: Story = {
  render: () => {
    const [expanded, setExpanded] = useState(false);
    const [selected, setSelected] = useState<string | null>(null);
    return (
      <ModelPickerGroup
        group={makeGroup()}
        isExpanded={expanded}
        isFavorite={false}
        selectedModelId={selected}
        canModelFit={fitAll}
        getModelFitStatus={fitStatus}
        onToggleExpand={() => setExpanded(!expanded)}
        onSelectModel={setSelected}
        onToggleFavorite={() => {}}
      />
    );
  },
};

export const SingleVariant: Story = {
  render: () => {
    const [selected, setSelected] = useState<string | null>(null);
    return (
      <ModelPickerGroup
        group={makeSingleGroup()}
        isExpanded={false}
        isFavorite={true}
        selectedModelId={selected}
        canModelFit={fitAll}
        getModelFitStatus={fitStatus}
        onToggleExpand={() => {}}
        onSelectModel={setSelected}
        onToggleFavorite={() => {}}
      />
    );
  },
};

export const TooLarge: Story = {
  args: {
    group: makeGroup({ name: 'DeepSeek V3 671B' }),
    isExpanded: false,
    isFavorite: false,
    selectedModelId: null,
    canModelFit: () => false,
    getModelFitStatus: () => 'too_large',
    onToggleExpand: () => {},
    onSelectModel: () => {},
    onToggleFavorite: () => {},
  },
};

export const WithInstanceRunning: Story = {
  render: () => {
    const [expanded, setExpanded] = useState(true);
    return (
      <ModelPickerGroup
        group={makeGroup()}
        isExpanded={expanded}
        isFavorite={false}
        selectedModelId="qwen3-30b-4bit"
        canModelFit={fitAll}
        getModelFitStatus={fitStatus}
        onToggleExpand={() => setExpanded(!expanded)}
        onSelectModel={() => {}}
        onToggleFavorite={() => {}}
        instanceStatuses={{ 'qwen3-30b-4bit': { status: 'Ready', statusClass: 'ready' } }}
      />
    );
  },
};

export const WithDownload: Story = {
  render: () => {
    const dlMap = new Map([['qwen3-30b-4bit', { available: true, nodeNames: ['Example workstation'], nodeIds: ['abc'] }]]);
    return (
      <ModelPickerGroup
        group={makeGroup()}
        isExpanded={false}
        isFavorite={false}
        selectedModelId={null}
        canModelFit={fitAll}
        getModelFitStatus={fitStatus}
        onToggleExpand={() => {}}
        onSelectModel={() => {}}
        onToggleFavorite={() => {}}
        downloadStatusMap={dlMap}
      />
    );
  },
};

export const Highlighted: Story = {
  args: {
    group: makeSingleGroup(),
    isExpanded: false,
    isFavorite: false,
    isHighlighted: true,
    selectedModelId: null,
    canModelFit: fitAll,
    getModelFitStatus: fitStatus,
    onToggleExpand: () => {},
    onSelectModel: () => {},
    onToggleFavorite: () => {},
  },
};

export const DownloadInProgress: Story = {
  args: { ...TooLarge.args, group: makeSingleGroup(), canModelFit: fitAll, getModelFitStatus: fitStatus,
    activeDownloads: [{ modelId: 'llama-8b-4bit', progress: .62, status: 'downloading' }], onCancelDownload: () => {} },
};
export const DownloadFailed: Story = {
  args: { ...DownloadInProgress.args, activeDownloads: [{ modelId: 'llama-8b-4bit', progress: .62, status: 'failed', error: 'The fixture transfer was interrupted. Retry the download.' }] },
};
export const LaunchAvailable: Story = {
  args: { ...DownloadInProgress.args, activeDownloads: [], downloadStatusMap: new Map([['llama-8b-4bit', { available: true, nodeNames: ['Example workstation'], nodeIds: ['fictional-node'] }]]), onLaunch: () => {} },
};
