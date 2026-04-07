import { useState } from 'react';
import styled, { css } from 'styled-components';
import { Button } from '../common/Button';

/* ================================================================
   Types
   ================================================================ */

export interface ImageGenerationParams {
  size: string;
  quality: 'low' | 'medium' | 'high';
  outputFormat: 'png' | 'jpeg';
  numImages: number;
  stream: boolean;
  partialImages: number;
  inputFidelity: 'low' | 'high';
  seed: number | null;
  numInferenceSteps: number | null;
  guidance: number | null;
  negativePrompt: string | null;
  numSyncSteps: number | null;
}

export const DEFAULT_IMAGE_PARAMS: ImageGenerationParams = {
  size: 'auto',
  quality: 'medium',
  outputFormat: 'png',
  numImages: 1,
  stream: false,
  partialImages: 0,
  inputFidelity: 'low',
  seed: null,
  numInferenceSteps: null,
  guidance: null,
  negativePrompt: null,
  numSyncSteps: null,
};

const SIZE_OPTIONS = ['auto', '512x512', '768x768', '1024x1024', '1024x768', '768x1024', '1024x1536', '1536x1024'];
const QUALITY_OPTIONS: Array<'low' | 'medium' | 'high'> = ['low', 'medium', 'high'];

export interface ImageParamsPanelProps {
  params: ImageGenerationParams;
  onChange: (params: ImageGenerationParams) => void;
  isEditMode?: boolean;
}

/* ================================================================
   Styles
   ================================================================ */

const Panel = styled.div`
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  font-size: ${({ theme }) => theme.fontSizes.label};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const ParamGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
`;

const Label = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

const Select = styled.select`
  all: unset;
  background: rgba(80, 80, 80, 0.5);
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 4px 8px;
  font: inherit;
  color: ${({ theme }) => theme.colors.text};
  cursor: pointer;

  &:focus {
    border-color: ${({ theme }) => theme.colors.goldDim};
    outline: none;
  }
`;

const ToggleGroup = styled.div`
  display: flex;
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  overflow: hidden;
`;

const ToggleBtn = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  padding: 4px 10px;
  font: inherit;
  transition: all 0.15s;

  ${({ $active }) =>
    $active
      ? css`
          background: ${({ theme }) => theme.colors.gold};
          color: ${({ theme }) => theme.colors.bg};
          font-weight: 600;
        `
      : css`
          background: rgba(80, 80, 80, 0.3);
          color: rgba(179, 179, 179, 0.8);
          &:hover { color: ${({ theme }) => theme.colors.text}; }
        `}
`;

const Switch = styled.button<{ $on: boolean }>`
  all: unset;
  cursor: pointer;
  width: 32px;
  height: 16px;
  border-radius: 8px;
  position: relative;
  transition: background 0.2s;

  ${({ $on }) =>
    $on
      ? css`background: ${({ theme }) => theme.colors.gold};`
      : css`background: rgba(80,80,80,0.5); border: 1px solid rgba(80,80,80,0.5);`}

  &::after {
    content: '';
    position: absolute;
    top: 2px;
    left: ${({ $on }) => ($on ? '16px' : '2px')};
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: ${({ $on }) => ($on ? '#000' : '#999')};
    transition: left 0.2s;
  }
`;

const NumberInput = styled.input`
  all: unset;
  width: 48px;
  text-align: center;
  background: rgba(80, 80, 80, 0.5);
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 4px 6px;
  font: inherit;
  color: ${({ theme }) => theme.colors.text};
  -moz-appearance: textfield;
  &::-webkit-outer-spin-button, &::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  &:focus { border-color: ${({ theme }) => theme.colors.goldDim}; outline: none; }
`;

const Divider = styled.div`
  width: 100%;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
  margin-top: 4px;
  padding-top: 8px;
`;

const AdvancedToggle = styled(Button)<{ $hasParams: boolean }>`
  ${({ $hasParams }) =>
    $hasParams &&
    css`
      &::after {
        content: '';
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: ${({ theme }) => theme.colors.gold};
      }
    `}
`;

const AdvancedSection = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const SliderRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
`;

const RangeInput = styled.input`
  flex: 1;
  -webkit-appearance: none;
  height: 4px;
  background: rgba(80, 80, 80, 0.5);
  border-radius: 2px;
  outline: none;

  &::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.gold};
    cursor: pointer;
  }

  &::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: ${({ theme }) => theme.colors.gold};
    cursor: pointer;
    border: none;
  }
`;


const TextArea = styled.textarea`
  all: unset;
  width: 100%;
  font: inherit;
  color: ${({ theme }) => theme.colors.text};
  background: rgba(80, 80, 80, 0.5);
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.sm};
  padding: 6px 8px;
  resize: none;
  box-sizing: border-box;
  &::placeholder { color: ${({ theme }) => theme.colors.textMuted}; }
  &:focus { border-color: ${({ theme }) => theme.colors.goldDim}; outline: none; }
`;

const ResetBtn = styled(Button)`
  align-self: flex-end;
`;

/* ================================================================
   Component
   ================================================================ */

export function ImageParamsPanel({ params, onChange, isEditMode = false }: ImageParamsPanelProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const update = (patch: Partial<ImageGenerationParams>) => onChange({ ...params, ...patch });

  const hasAdvancedParams =
    params.seed !== null ||
    params.numInferenceSteps !== null ||
    params.guidance !== null ||
    (params.negativePrompt !== null && params.negativePrompt !== '') ||
    params.numSyncSteps !== null;

  return (
    <Panel>
      {/* Size */}
      <ParamGroup>
        <Label>Size</Label>
        <Select value={params.size} onChange={(e) => update({ size: e.target.value })}>
          {SIZE_OPTIONS.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </Select>
      </ParamGroup>

      {/* Quality */}
      <ParamGroup>
        <Label>Quality</Label>
        <ToggleGroup>
          {QUALITY_OPTIONS.map((q) => (
            <ToggleBtn key={q} $active={params.quality === q} onClick={() => update({ quality: q })}>
              {q}
            </ToggleBtn>
          ))}
        </ToggleGroup>
      </ParamGroup>

      {/* Format */}
      <ParamGroup>
        <Label>Format</Label>
        <ToggleGroup>
          {(['png', 'jpeg'] as const).map((f) => (
            <ToggleBtn key={f} $active={params.outputFormat === f} onClick={() => update({ outputFormat: f })}>
              {f}
            </ToggleBtn>
          ))}
        </ToggleGroup>
      </ParamGroup>

      {/* Num images or Input Fidelity */}
      {isEditMode ? (
        <ParamGroup>
          <Label>Fidelity</Label>
          <ToggleGroup>
            {(['low', 'high'] as const).map((f) => (
              <ToggleBtn key={f} $active={params.inputFidelity === f} onClick={() => update({ inputFidelity: f })}>
                {f}
              </ToggleBtn>
            ))}
          </ToggleGroup>
        </ParamGroup>
      ) : (
        <ParamGroup>
          <Label>Images</Label>
          <NumberInput
            type="number"
            min={1}
            value={params.numImages}
            onChange={(e) => update({ numImages: Math.max(1, parseInt(e.target.value) || 1) })}
          />
        </ParamGroup>
      )}

      {/* Stream toggle */}
      {!isEditMode && (
        <ParamGroup>
          <Label>Stream</Label>
          <Switch $on={params.stream} onClick={() => update({ stream: !params.stream })} />
          {params.stream && (
            <>
              <Label>Partial</Label>
              <NumberInput
                type="number"
                min={0}
                value={params.partialImages}
                onChange={(e) => update({ partialImages: Math.max(0, parseInt(e.target.value) || 0) })}
              />
            </>
          )}
        </ParamGroup>
      )}

      {/* Advanced section */}
      <Divider />
      <AdvancedToggle
        variant="ghost"
        size="sm"
        $hasParams={hasAdvancedParams && !showAdvanced}
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? '▾' : '▸'} Advanced
      </AdvancedToggle>

      {showAdvanced && (
        <AdvancedSection>
          {/* Seed + Steps */}
          <ParamGroup style={{ flexWrap: 'wrap' }}>
            <Label>Seed</Label>
            <NumberInput
              type="number"
              min={0}
              value={params.seed ?? ''}
              onChange={(e) => update({ seed: e.target.value === '' ? null : parseInt(e.target.value) })}
              placeholder="—"
            />
            <Label>Steps</Label>
            <SliderRow>
              <RangeInput
                type="range" min={1} max={100}
                value={params.numInferenceSteps ?? 50}
                onChange={(e) => update({ numInferenceSteps: parseInt(e.target.value) })}
              />
              <span>{params.numInferenceSteps ?? '—'}</span>
              {params.numInferenceSteps !== null && (
                <Button variant="ghost" size="sm" onClick={() => update({ numInferenceSteps: null })}>✕</Button>
              )}
            </SliderRow>
          </ParamGroup>

          {/* Guidance */}
          <ParamGroup>
            <Label>Guidance</Label>
            <SliderRow>
              <RangeInput
                type="range" min={1} max={20} step={0.5}
                value={params.guidance ?? 7.5}
                onChange={(e) => update({ guidance: parseFloat(e.target.value) })}
              />
              <span>{params.guidance?.toFixed(1) ?? '—'}</span>
              {params.guidance !== null && (
                <Button variant="ghost" size="sm" onClick={() => update({ guidance: null })}>✕</Button>
              )}
            </SliderRow>
          </ParamGroup>

          {/* Negative prompt */}
          <div style={{ width: '100%' }}>
            <Label>Negative prompt</Label>
            <TextArea
              rows={2}
              placeholder="Things to avoid…"
              value={params.negativePrompt ?? ''}
              onChange={(e) => update({ negativePrompt: e.target.value || null })}
            />
          </div>

          {/* Reset */}
          <ResetBtn variant="ghost" size="sm" onClick={() => {
            onChange({ ...DEFAULT_IMAGE_PARAMS });
            setShowAdvanced(false);
          }}>
            Reset all
          </ResetBtn>
        </AdvancedSection>
      )}
    </Panel>
  );
}
