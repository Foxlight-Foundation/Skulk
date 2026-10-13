import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { ImageParamsPanel, DEFAULT_IMAGE_PARAMS, type ImageGenerationParams } from './ImageParamsPanel';

const meta: Meta<typeof ImageParamsPanel> = {
  title: 'Display/ImageParamsPanel',
  component: ImageParamsPanel,
  parameters: { layout: 'centered' },
  decorators: [
    (Story) => (
      <div style={{ width: 'min(600px, calc(100vw - 48px))', padding: 24, }}>
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof ImageParamsPanel>;

export const Default: Story = {
  render: () => {
    const [params, setParams] = useState<ImageGenerationParams>({ ...DEFAULT_IMAGE_PARAMS });
    return <ImageParamsPanel params={params} onChange={setParams} />;
  },
};

export const EditMode: Story = {
  render: () => {
    const [params, setParams] = useState<ImageGenerationParams>({ ...DEFAULT_IMAGE_PARAMS });
    return <ImageParamsPanel params={params} onChange={setParams} isEditMode />;
  },
};

export const WithAdvanced: Story = {
  render: () => {
    const [params, setParams] = useState<ImageGenerationParams>({
      ...DEFAULT_IMAGE_PARAMS,
      seed: 42,
      numInferenceSteps: 30,
      guidance: 7.5,
      negativePrompt: 'blurry, low quality',
    });
    return <ImageParamsPanel params={params} onChange={setParams} />;
  },
};


export const QwenTurbo: Story = {
  render: () => {
    const [params, setParams] = useState<ImageGenerationParams>({ ...DEFAULT_IMAGE_PARAMS });
    return <ImageParamsPanel params={params} onChange={setParams} imageProfile={{
      family: 'qwen_image_21_turbo', defaultWidth: 2048, defaultHeight: 2048,
      dimensionMultiple: 32, maxPixels: 4400000, maxReferenceImages: 10,
      supportsAlpha: true, defaultSteps: 8, defaultGuidance: 1,
      fixedSigmas: [1, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0],
    }} />;
  },
};
