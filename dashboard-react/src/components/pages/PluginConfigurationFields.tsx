import { Select as DesignedSelect } from '../common/Select';
import { useId } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { Button } from '../common/Button';
import { fieldLabel, resolveSchema, schemaKind, schemaRecord, settingsField } from './pluginSettingsSchema';

const Fields = styled.div`display: grid; gap: 20px; min-width: 0;`;
const Field = styled.div`display: grid; gap: 6px; min-width: 0;`;
const LabelRow = styled.div`
  display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
  label { font-size: ${({ theme }) => theme.fontSizes.sm}; font-weight: 600; color: ${({ theme }) => theme.colors.text}; overflow-wrap: anywhere; }
`;
const Required = styled.span`font-size: 12px; color: ${({ theme }) => theme.colors.metadataText};`;
const Help = styled.p`margin: 0; font-size: 12.5px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary}; overflow-wrap: anywhere;`;
const Control = styled.div`display: flex; align-items: center; gap: 8px; min-width: 0; margin-top: 2px; > :first-child { flex: 1 1 auto; min-width: 0; }`;
const Input = styled.input`
  box-sizing: border-box; width: 100%; min-height: 36px; padding: 8px 11px; border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ theme }) => theme.colors.borderControl};
  background: ${({ theme }) => theme.colors.surface}; color: ${({ theme }) => theme.colors.text};
  font: 14px ${({ theme }) => theme.fonts.body};
  &::placeholder { color: ${({ theme }) => theme.colors.textMuted}; }
  &:focus-visible { outline: none; box-shadow: ${({ theme }) => theme.colors.focusRing}; }
  &:disabled { opacity: .55; cursor: not-allowed; }
`;
const Select = styled(DesignedSelect)`width: 100%; min-height: 36px; font-size: 14px;`;
const Group = styled.fieldset`
  margin: 0; padding: 14px 16px 16px; min-width: 0; display: grid; gap: 14px;
  border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: ${({ theme }) => theme.radii.lg};
  legend { padding: 0 6px; font-size: ${({ theme }) => theme.fontSizes.sm}; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
`;

/** Props shared by recursive ordinary-setting controls; secrets are excluded. */
interface ConfigurationFieldsProps {
  /** The object schema whose properties become fields. */
  schema: Record<string, unknown>;
  /** The document local `#/` references resolve against; the schema itself at the top level. */
  rootSchema?: Record<string, unknown>;
  /** The draft values; an omitted key leaves the plugin's own default in force. */
  values: Record<string, unknown>;
  /** Called with the whole next draft after any change. */
  onChange: (values: Record<string, unknown>) => void;
  disabled?: boolean;
}

/**
 * Render plugin-declared scalar and nested-object controls without provider code.
 *
 * Each field shows its label, the schema's description as help text, and its
 * control. A key the draft omits shows the schema default as a placeholder,
 * since the plugin applies that default itself. A nullable scalar (an
 * optional value) submits `null` when its input is emptied. Callers check
 * `supportedConfigurationSchema` first; this renders only what it accepts.
 */
export function PluginConfigurationFields({ schema, rootSchema = schema, values, onChange, disabled }: ConfigurationFieldsProps) {
  const id = useId();
  const { t } = useSkulkTranslation();
  const resolved = resolveSchema(schema, rootSchema);
  const required = Array.isArray(resolved.required) ? resolved.required : [];
  return <Fields>{Object.entries(schemaRecord(resolved.properties)).map(([name, raw]) => {
    const { schema: field, nullable } = settingsField(schemaRecord(raw), rootSchema);
    const type = schemaKind(field);
    const label = fieldLabel(name, field.title);
    const fieldId = `${id}-${name}`;
    const helpId = `${fieldId}-help`;
    const value = values[name];
    const isRequired = required.includes(name);
    const description = typeof field.description === 'string' && field.description.trim() ? field.description.trim() : null;
    // The plugin applies its own default while the key is omitted.
    const fallback = field.default;
    const hasDefault = fallback !== undefined && fallback !== null;
    // Emptying an optional value clears it to null; anything else is left out.
    const cleared = nullable ? null : undefined;
    const update = (next: unknown) => {
      const copy = { ...values };
      if (next === undefined) delete copy[name]; else Object.defineProperty(copy, name, { value: next, writable: true, enumerable: true, configurable: true });
      onChange(copy);
    };
    if (type === 'object') return <Group key={name} disabled={disabled}><legend>{label}</legend>
      {description ? <Help>{description}</Help> : null}
      <PluginConfigurationFields schema={field} rootSchema={rootSchema} values={schemaRecord(value)} onChange={update} disabled={disabled} />
    </Group>;
    const display = (option: unknown) => option === true ? t('plugins.yes', 'Yes') : option === false ? t('plugins.no', 'No') : String(option);
    const unsetLabel = hasDefault ? t('plugins.defaultOption', 'Default ({value})', { value: display(fallback) }) : t('plugins.unset', 'Not set');
    const placeholder = (value === undefined || value === null) && hasDefault ? String(fallback)
      : nullable && (value === undefined || value === null) ? t('plugins.optionalValue', 'Optional') : undefined;
    // Returning to the default is offered only when it changes something.
    const resettable = !isRequired && value !== undefined && !(value === null && !hasDefault);
    const describedBy = description ? helpId : undefined;
    const control = Array.isArray(field.enum)
      ? <Select id={fieldId} aria-describedby={describedBy} value={value === undefined || value === null ? '' : JSON.stringify(value)} disabled={disabled} required={isRequired}
        onValueChange={(selectedValue) => update(selectedValue === '' ? cleared : JSON.parse(selectedValue))}>
        <option value="">{unsetLabel}</option>
        {field.enum.filter((option) => option !== null).map((option) => <option key={JSON.stringify(option)} value={JSON.stringify(option)}>{display(option)}</option>)}
      </Select>
      : type === 'boolean'
        ? <Select id={fieldId} aria-describedby={describedBy} value={value === undefined || value === null ? '' : String(value)} disabled={disabled} required={isRequired}
          onValueChange={(selectedValue) => update(selectedValue === '' ? cleared : selectedValue === 'true')}>
          <option value="">{unsetLabel}</option><option value="true">{t('plugins.yes', 'Yes')}</option><option value="false">{t('plugins.no', 'No')}</option>
        </Select>
        : <Input id={fieldId} aria-describedby={describedBy} type={type === 'string' ? 'text' : 'number'} value={typeof value === 'string' || typeof value === 'number' ? value : ''} disabled={disabled}
          required={isRequired} step={type === 'integer' ? 1 : 'any'} placeholder={placeholder}
          min={typeof field.minimum === 'number' ? field.minimum : undefined} max={typeof field.maximum === 'number' ? field.maximum : undefined}
          minLength={typeof field.minLength === 'number' ? field.minLength : undefined} maxLength={typeof field.maxLength === 'number' ? field.maxLength : undefined}
          onChange={(event) => update(type === 'string'
            ? (event.target.value === '' && nullable ? null : event.target.value)
            : Number.isFinite(event.target.valueAsNumber) ? event.target.valueAsNumber : cleared)} />;
    return <Field key={name}>
      <LabelRow><label htmlFor={fieldId}>{label}</label>{isRequired ? <Required>{t('plugins.required', 'Required')}</Required> : null}</LabelRow>
      {description ? <Help id={helpId}>{description}</Help> : null}
      <Control>
        {control}
        {resettable ? <Button type="button" variant="outline" size="sm" disabled={disabled} onClick={() => update(undefined)}>{hasDefault ? t('plugins.useDefault', 'Use default') : t('plugins.omit', 'Leave unset')}</Button> : null}
      </Control>
    </Field>;
  })}</Fields>;
}
