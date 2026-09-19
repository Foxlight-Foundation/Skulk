import type { Meta, StoryObj } from '@storybook/react-vite';
import { expect, userEvent, within } from 'storybook/test';
import { App } from './App';

const meta = {
  title: 'Screens/Dashboard',
  component: App,
  beforeEach: () => {
    // The component test runner uses `/`; prevent App's real bookmark loader
    // from overriding the story's route with Cluster.
    const previous = location.href;
    history.replaceState(null, '', `/iframe.html${location.search}`);
    return () => history.replaceState(null, '', previous);
  },
  parameters: { layout: 'fullscreen', screenRoute: 'cluster' },
  decorators: [(Story) => <div style={{ height: '100dvh', isolation: 'isolate' }}><Story /></div>],
} satisfies Meta<typeof App>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Cluster: Story = {};
export const Integrations: Story = { parameters: { screenRoute: 'integrations' } };
export const Plugins: Story = { parameters: { screenRoute: 'plugins' } };
export const Chat: Story = { parameters: { screenRoute: 'chat' } };
export const ModelStore: Story = { parameters: { screenRoute: 'model-store' } };
export const Settings: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    if (!canvas.queryByRole('button', { name: 'Settings' })) await userEvent.click(await canvas.findByRole('button', { name: 'Toggle mobile menu' }));
    await userEvent.click(await canvas.findByRole('button', { name: 'Settings' }));
    await expect(within(document.body).getByRole('dialog')).toBeVisible();
  },
};
export const Steward: Story = {
  play: async () => {
    await userEvent.keyboard('{Control>}k{/Control}');
    await expect(await within(document.body).findByRole('dialog', { name: 'Skulk Steward' })).toBeVisible();
  },
};

export const FindModels: Story = {
  parameters: { screenRoute: 'model-store' },
  play: async ({ canvasElement }) => {
    await userEvent.click(await within(canvasElement).findByRole('button', { name: 'Find Models' }));
    await expect(await within(document.body).findByRole('dialog')).toBeVisible();
  },
};
export const Devices: Story = {
  play: async context => {
    await Settings.play?.(context);
    await userEvent.click(await within(document.body).findByRole('button', { name: 'Devices & pairing' }));
    await expect(within(document.body).getByRole('dialog', { name: 'Devices & pairing' })).toBeVisible();
  },
};
export const IntegrationSetup: Story = {
  parameters: { screenRoute: 'integrations' },
  play: async ({ canvasElement }) => {
    await userEvent.click(await within(canvasElement).findByRole('button', { name: 'Claude Code' }));
    await expect(await within(document.body).findByRole('dialog')).toBeVisible();
  },
};
export const StewardConversation: Story = {
  play: async context => {
    await Steward.play?.(context);
    const body = within(document.body);
    await userEvent.type(await body.findByPlaceholderText('Ask about the cluster...'), 'How is this cluster doing?');
    await userEvent.keyboard('{Enter}');
    await expect(await body.findByText(/The example cluster has three nodes/)).toBeVisible();
  },
};

export const StewardContinuity: Story = {
  play: async context => {
    await StewardConversation.play?.(context);
    const body = within(document.body);
    const draft = 'Keep this draft while changing views';
    await userEvent.type(body.getByPlaceholderText('Ask about the cluster...'), draft);
    await userEvent.click(body.getByRole('button', { name: 'Open as page' }));
    await expect(body.queryByRole('dialog')).not.toBeInTheDocument();
    await expect(body.getByPlaceholderText('Ask about the cluster...')).toHaveValue(draft);
    await userEvent.keyboard('{Control>}k{/Control}');
    await expect(body.getByRole('dialog', { name: 'Skulk Steward' })).toBeVisible();
    await expect(body.getByPlaceholderText('Ask about the cluster...')).toHaveValue(draft);
    await expect(body.getByText(/The example cluster has three nodes/)).toBeVisible();
  },
};
