/**
 * ImageParamsPanel
 *
 * Controls for image generation parameters: size, quality, format,
 * stream toggle, and advanced options (seed, steps, guidance, etc.).
 * Ported from ImageParamsPanel.svelte.
 */
import React, { useRef, useState } from 'react';
import styled, { createGlobalStyle } from 'styled-components';
import { useChatStore, type ImageGenerationParams } from '../../stores/chatStore';

// ─── Global range slider thumb styles ─────────────────────────────────────────

const RangeStyles = createGlobalStyle`
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: oklch(0.85 0.18 85);
    cursor: pointer;
    border: none;
  }
  input[type="range"]::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: oklch(0.85 0.18 85);
    cursor: pointer;
    border: none;
  }
  input[type="number"]::-webkit-inner-spin-button,
  input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
  input[type="number"] { -moz-appearance: textfield; }
`;

// ─── Styled components ────────────────────────────────────────────────────────

const Panel = styled.div`
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  padding: 8px 12px;
`;

const BasicRow = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
`;

const FieldGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
`;

const Label = styled.span`
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  text-transform: uppercase;
  letter-spacing: 0.08em;
  white-space: nowrap;
`;

const DropdownBtn = styled.button`
  background: ${({ theme }) => theme.colors.mediumGray};
  border: 1px solid ${({ theme }) => theme.colors.yellowDarker};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 3px 20px 3px 8px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.yellow};
  cursor: pointer;
  position: relative;
  transition: border-color ${({ theme }) => theme.transitions.fast};
  &:hover { border-color: ${({ theme }) => theme.colors.yellow}; }
  &:focus { outline: none; border-color: ${({ theme }) => theme.colors.yellow}; }
`;

const DropdownArrow = styled.span<{ $open: boolean }>`
  position: absolute;
  right: 6px;
  top: 50%;
  transform: translateY(-50%) ${({ $open }) => ($open ? 'rotate(180deg)' : 'rotate(0deg)')};
  font-size: 8px;
  color: ${({ theme }) => theme.colors.yellow};
  transition: transform ${({ theme }) => theme.transitions.fast};
  pointer-events: none;
`;

const DropdownMenu = styled.div`
  position: fixed;
  background: ${({ theme }) => theme.colors.darkGray};
  border: 1px solid ${({ theme }) => theme.colors.yellowDarker};
  border-radius: ${({ theme }) => theme.radius.sm};
  box-shadow: 0 8px 24px oklch(0 0 0 / 0.5);
  z-index: 9999;
  max-height: 192px;
  overflow-y: auto;
  min-width: max-content;
`;

const DropdownItem = styled.button<{ $active: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 5px 12px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-align: left;
  background: none;
  border: none;
  cursor: pointer;
  color: ${({ theme, $active }) => ($active ? theme.colors.yellow : theme.colors.lightGray)};
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

const SegmentedGroup = styled.div`
  display: flex;
  border: 1px solid ${({ theme }) => theme.colors.yellowDarker};
  border-radius: ${({ theme }) => theme.radius.sm};
  overflow: hidden;
`;

const SegmentBtn = styled.button<{ $active: boolean }>`
  padding: 3px 8px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  cursor: pointer;
  border: none;
  transition: all ${({ theme }) => theme.transitions.fast};
  background: ${({ theme, $active }) => ($active ? theme.colors.yellow : `${theme.colors.mediumGray}80`)};
  color: ${({ theme, $active }) => ($active ? theme.colors.black : theme.colors.lightGray)};
  &:hover { color: ${({ theme, $active }) => ($active ? theme.colors.black : theme.colors.yellow)}; }
`;

const NumberInput = styled.input`
  width: 48px;
  background: ${({ theme }) => theme.colors.mediumGray};
  border: 1px solid ${({ theme }) => theme.colors.yellowDarker};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 3px 6px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.yellow};
  text-align: center;
  transition: border-color ${({ theme }) => theme.transitions.fast};
  &:hover { border-color: ${({ theme }) => theme.colors.yellow}; }
  &:focus { outline: none; border-color: ${({ theme }) => theme.colors.yellow}; }
`;

const ToggleTrack = styled.button<{ $on: boolean }>`
  width: 32px;
  height: 16px;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  position: relative;
  transition: all ${({ theme }) => theme.transitions.fast};
  background: ${({ theme, $on }) => ($on ? theme.colors.yellow : `${theme.colors.mediumGray}80`)};
  outline: 1px solid ${({ theme, $on }) => ($on ? theme.colors.yellow : theme.colors.border)};
`;

const ToggleThumb = styled.div<{ $on: boolean }>`
  position: absolute;
  top: 2px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  transition: all ${({ theme }) => theme.transitions.fast};
  background: ${({ theme, $on }) => ($on ? theme.colors.black : theme.colors.lightGray)};
  ${({ $on }) => ($on ? 'right: 2px;' : 'left: 2px;')}
`;

const AdvancedToggle = styled.button<{ $active: boolean }>`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: none;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  color: ${({ theme, $active }) => ($active ? theme.colors.yellow : theme.colors.lightGray)};
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

const AdvancedSection = styled.div`
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid ${({ theme }) => theme.colors.border};
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const AdvRow = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
`;

const RangeInput = styled.input`
  flex: 1;
  height: 4px;
  background: ${({ theme }) => theme.colors.mediumGray};
  border-radius: 2px;
  appearance: none;
  cursor: pointer;
  accent-color: ${({ theme }) => theme.colors.yellow};
`;

const RangeValue = styled.span`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.yellow};
  min-width: 32px;
  text-align: right;
`;

const ClearBtn = styled.button`
  background: none;
  border: none;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.lightGray};
  padding: 2px;
  display: flex;
  align-items: center;
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

const NegativeTextarea = styled.textarea`
  width: 100%;
  background: ${({ theme }) => theme.colors.mediumGray};
  border: 1px solid ${({ theme }) => theme.colors.yellowDarker};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 6px 8px;
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: ${({ theme }) => theme.colors.yellow};
  resize: none;
  transition: border-color ${({ theme }) => theme.transitions.fast};
  &::placeholder { color: ${({ theme }) => theme.colors.mutedForeground}; }
  &:hover { border-color: ${({ theme }) => theme.colors.yellow}; }
  &:focus { outline: none; border-color: ${({ theme }) => theme.colors.yellow}; }
`;

const Spacer = styled.div`flex: 1;`;

const ResetBtn = styled.button`
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: none;
  border: none;
  cursor: pointer;
  color: ${({ theme }) => theme.colors.lightGray};
  align-self: flex-end;
  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
`;

// ─── Dropdown component ────────────────────────────────────────────────────────

interface DropdownProps<T extends string> {
  value: T;
  options: T[];
  onChange: (v: T) => void;
}

function Dropdown<T extends string>({ value, options, onChange }: DropdownProps<T>) {
  const [open, setOpen] = useState(false);
  const btnRef = useRef<HTMLButtonElement>(null);

  const rect = btnRef.current?.getBoundingClientRect();

  return (
    <div style={{ position: 'relative' }}>
      <DropdownBtn
        ref={btnRef}
        type="button"
        onClick={() => setOpen((p) => !p)}
      >
        {value.toUpperCase()}
        <DropdownArrow $open={open}>▼</DropdownArrow>
      </DropdownBtn>
      {open && (
        <>
          <div
            style={{ position: 'fixed', inset: 0, zIndex: 9998 }}
            onClick={() => setOpen(false)}
          />
          <DropdownMenu
            style={
              rect
                ? {
                    bottom: `calc(100vh - ${rect.top}px + 4px)`,
                    left: `${rect.left}px`,
                  }
                : undefined
            }
          >
            {options.map((opt) => (
              <DropdownItem
                key={opt}
                $active={opt === value}
                onClick={() => { onChange(opt); setOpen(false); }}
              >
                {opt === value ? '✓ ' : '  '}
                {opt.toUpperCase()}
              </DropdownItem>
            ))}
          </DropdownMenu>
        </>
      )}
    </div>
  );
}

// ─── Component ─────────────────────────────────────────────────────────────────

export interface ImageParamsPanelProps {
  isEditMode?: boolean;
}

const SIZE_OPTIONS: ImageGenerationParams['size'][] = [
  'auto', '512x512', '768x768', '1024x1024',
  '1024x768', '768x1024', '1024x1536', '1536x1024',
];
const QUALITY_OPTIONS: ImageGenerationParams['quality'][] = ['low', 'medium', 'high'];
const FORMAT_OPTIONS: ImageGenerationParams['outputFormat'][] = ['png', 'jpeg'];
const FIDELITY_OPTIONS: ImageGenerationParams['inputFidelity'][] = ['low', 'high'];

export const ImageParamsPanel: React.FC<ImageParamsPanelProps> = ({ isEditMode = false }) => {
  const params = useChatStore((s) => s.imageGenerationParams);
  const setParams = useChatStore((s) => s.setImageGenerationParams);
  const resetParams = useChatStore((s) => s.resetImageGenerationParams);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const hasAdvancedParams =
    params.seed !== null ||
    params.numInferenceSteps !== null ||
    params.guidance !== null ||
    (params.negativePrompt !== null && params.negativePrompt.trim() !== '') ||
    params.numSyncSteps !== null;

  return (
    <>
      <RangeStyles />
      <Panel>
        <BasicRow>
          {/* Size */}
          <FieldGroup>
            <Label>Size:</Label>
            <Dropdown<ImageGenerationParams['size']>
              value={params.size}
              options={SIZE_OPTIONS}
              onChange={(v) => setParams({ size: v })}
            />
          </FieldGroup>

          {/* Quality */}
          <FieldGroup>
            <Label>Quality:</Label>
            <Dropdown<ImageGenerationParams['quality']>
              value={params.quality}
              options={QUALITY_OPTIONS}
              onChange={(v) => setParams({ quality: v })}
            />
          </FieldGroup>

          {/* Format */}
          <FieldGroup>
            <Label>Format:</Label>
            <SegmentedGroup>
              {FORMAT_OPTIONS.map((fmt) => (
                <SegmentBtn
                  key={fmt}
                  type="button"
                  $active={params.outputFormat === fmt}
                  onClick={() => setParams({ outputFormat: fmt })}
                >
                  {fmt}
                </SegmentBtn>
              ))}
            </SegmentedGroup>
          </FieldGroup>

          {/* Number of images (not in edit mode) */}
          {!isEditMode && (
            <FieldGroup>
              <Label>Images:</Label>
              <NumberInput
                type="number"
                min={1}
                value={params.numImages}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v) && v >= 1) setParams({ numImages: v });
                }}
              />
            </FieldGroup>
          )}

          {/* Stream toggle */}
          <FieldGroup>
            <Label>Stream:</Label>
            <ToggleTrack
              type="button"
              $on={params.stream}
              onClick={() => setParams({ stream: !params.stream })}
            >
              <ToggleThumb $on={params.stream} />
            </ToggleTrack>
          </FieldGroup>

          {/* Partial images (stream only) */}
          {params.stream && (
            <FieldGroup>
              <Label>Partials:</Label>
              <NumberInput
                type="number"
                min={0}
                value={params.partialImages}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  if (!isNaN(v) && v >= 0) setParams({ partialImages: v });
                }}
              />
            </FieldGroup>
          )}

          {/* Fidelity (edit mode only) */}
          {isEditMode && (
            <FieldGroup>
              <Label>Fidelity:</Label>
              <SegmentedGroup>
                {FIDELITY_OPTIONS.map((f) => (
                  <SegmentBtn
                    key={f}
                    type="button"
                    $active={params.inputFidelity === f}
                    onClick={() => setParams({ inputFidelity: f })}
                    title={f === 'low' ? 'More creative variation' : 'Closer to original'}
                  >
                    {f}
                  </SegmentBtn>
                ))}
              </SegmentedGroup>
            </FieldGroup>
          )}

          <Spacer />

          {/* Advanced toggle */}
          <AdvancedToggle
            type="button"
            $active={showAdvanced || hasAdvancedParams}
            onClick={() => setShowAdvanced((p) => !p)}
          >
            <span>Advanced</span>
            <span style={{ fontSize: 8, transition: 'transform 0.2s', display: 'inline-block', transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)' }}>▼</span>
            {hasAdvancedParams && !showAdvanced && (
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'oklch(0.85 0.18 85)', display: 'inline-block' }} />
            )}
          </AdvancedToggle>
        </BasicRow>

        {/* Advanced section */}
        {showAdvanced && (
          <AdvancedSection>
            {/* Row 1: Seed + Steps */}
            <AdvRow>
              <Label>Seed:</Label>
              <NumberInput
                type="number"
                min={0}
                style={{ width: 80 }}
                value={params.seed ?? ''}
                placeholder="Random"
                onChange={(e) => {
                  const v = e.target.value.trim();
                  if (v === '') { setParams({ seed: null }); return; }
                  const n = parseInt(v, 10);
                  if (!isNaN(n) && n >= 0) setParams({ seed: n });
                }}
              />
              <Label style={{ marginLeft: 8 }}>Steps:</Label>
              <RangeInput
                type="range"
                min={1}
                max={100}
                value={params.numInferenceSteps ?? 50}
                onChange={(e) => setParams({ numInferenceSteps: parseInt(e.target.value, 10) })}
              />
              <RangeValue>{params.numInferenceSteps ?? '--'}</RangeValue>
              {params.numInferenceSteps !== null && (
                <ClearBtn type="button" onClick={() => setParams({ numInferenceSteps: null })} title="Clear">✕</ClearBtn>
              )}
            </AdvRow>

            {/* Row 2: Guidance */}
            <AdvRow>
              <Label>Guidance:</Label>
              <RangeInput
                type="range"
                min={1}
                max={20}
                step={0.5}
                value={params.guidance ?? 7.5}
                onChange={(e) => setParams({ guidance: parseFloat(e.target.value) })}
              />
              <RangeValue>{params.guidance !== null ? params.guidance.toFixed(1) : '--'}</RangeValue>
              {params.guidance !== null && (
                <ClearBtn type="button" onClick={() => setParams({ guidance: null })} title="Clear">✕</ClearBtn>
              )}
            </AdvRow>

            {/* Row 3: Sync steps */}
            <AdvRow>
              <Label>Sync Steps:</Label>
              <RangeInput
                type="range"
                min={1}
                max={100}
                value={params.numSyncSteps ?? 1}
                onChange={(e) => setParams({ numSyncSteps: parseInt(e.target.value, 10) })}
              />
              <RangeValue>{params.numSyncSteps ?? '--'}</RangeValue>
              {params.numSyncSteps !== null && (
                <ClearBtn type="button" onClick={() => setParams({ numSyncSteps: null })} title="Clear">✕</ClearBtn>
              )}
            </AdvRow>

            {/* Row 4: Negative prompt */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <Label>Negative Prompt:</Label>
              <NegativeTextarea
                rows={2}
                value={params.negativePrompt ?? ''}
                placeholder="Things to avoid in the image..."
                onChange={(e) => setParams({ negativePrompt: e.target.value || null })}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: 4 }}>
              <ResetBtn
                type="button"
                onClick={() => { resetParams(); setShowAdvanced(false); }}
              >
                Reset to Defaults
              </ResetBtn>
            </div>
          </AdvancedSection>
        )}
      </Panel>
    </>
  );
};
