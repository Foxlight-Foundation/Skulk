import { Select as DesignedSelect } from '../common/Select';
import { useId, useState, type ComponentProps } from 'react';
import styled from 'styled-components';
import { useSkulkTranslation } from '../../i18n/tolgee';
import { Button } from '../common/Button';
import {
  fieldLabel, formatUsdAmount, isAdvancedField, isWholeCents, resolveSchema, schemaKind, schemaRecord, settingsField, settingsUnit, type SettingsField, type SettingsUnit,
} from './pluginSettingsSchema';

const Fields = styled.div`display: grid; gap: 20px; min-width: 0;`;
const Field = styled.div`display: grid; gap: 6px; min-width: 0;`;
const LabelRow = styled.div`
  display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
  label { font-size: ${({ theme }) => theme.fontSizes.sm}; font-weight: 600; color: ${({ theme }) => theme.colors.text}; overflow-wrap: anywhere; }
`;
const Required = styled.span`font-size: 12px; color: ${({ theme }) => theme.colors.metadataText};`;
const Help = styled.p`margin: 0; font-size: 12.5px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary}; overflow-wrap: anywhere;`;
const Control = styled.div`display: flex; align-items: center; gap: 8px; min-width: 0; margin-top: 2px; > :first-child { flex: 1 1 auto; min-width: 0; }`;
const Affixed = styled.div`display: flex; align-items: center; gap: 8px; min-width: 0; > input { flex: 1 1 auto; min-width: 0; }`;
const UnitText = styled.span`flex: none; font-size: 14px; color: ${({ theme }) => theme.colors.textSecondary};`;
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
const Advanced = styled.details`
  min-width: 0; border-top: 1px solid ${({ theme }) => theme.colors.border}; padding-top: 14px;
  > summary {
    cursor: pointer; width: fit-content; border-radius: ${({ theme }) => theme.radii.md};
    font-size: ${({ theme }) => theme.fontSizes.sm}; font-weight: 600; color: ${({ theme }) => theme.colors.textSecondary};
    &:focus-visible { outline: none; box-shadow: ${({ theme }) => theme.colors.focusRing}; }
  }
  &[open] > summary { margin-bottom: 16px; }
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

/** Props for a number input that shows a saved value in its unit's own format. */
interface FormattedNumberInputProps extends Omit<ComponentProps<typeof Input>, 'value' | 'onChange' | 'type'> {
  /** The saved or drafted value; anything but a number shows an empty input. */
  value: unknown;
  /** How a value reads while the field is not being edited, such as "10.00" for dollars. */
  format: (value: number) => string;
  /** Called with the number typed, or null when the input no longer holds one. */
  onNumber: (value: number | null) => void;
}

/**
 * A number input that shows its value formatted ("10.00") until it is edited.
 *
 * While the person types, the text they typed stays exactly as typed, so a
 * reformat never moves the cursor or rewrites "10." into "10.00" mid-entry;
 * leaving the field shows the formatted value again.
 */
function FormattedNumberInput({ value, format, onNumber, ...props }: FormattedNumberInputProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (typeof value === 'number' && Number.isFinite(value) ? format(value) : '');
  return <Input {...props} type="number" value={shown}
    onChange={(event) => { setDraft(event.target.value); onNumber(Number.isFinite(event.target.valueAsNumber) ? event.target.valueAsNumber : null); }}
    onBlur={() => setDraft(null)} />;
}

/**
 * Render plugin-declared scalar and nested-object controls without provider code.
 *
 * Each field shows its label, the schema's description as help text, and its
 * control. A key the draft omits shows the schema default as a placeholder,
 * since the plugin applies that default itself. A nullable scalar (an
 * optional value) submits `null` when its input is emptied. Callers check
 * `supportedConfigurationSchema` first; this renders only what it accepts.
 *
 * Two optional field hints come from the plugin's schema. `"x-skulk-advanced":
 * true` moves a field into a collapsed Advanced section after the others,
 * which opens by itself when one of its required fields has no value.
 * `"x-skulk-unit"` (`usd`, `minutes` or `hours`) shows a number field's unit
 * beside its control; the value is stored in that unit and never converted.
 */
export function PluginConfigurationFields({ schema, rootSchema = schema, values, onChange, disabled }: ConfigurationFieldsProps) {
  const id = useId();
  const { t } = useSkulkTranslation();
  const resolved = resolveSchema(schema, rootSchema);
  const required = Array.isArray(resolved.required) ? resolved.required : [];
  const fields = Object.entries(schemaRecord(resolved.properties)).map(([name, raw]) => ({ name, ...settingsField(schemaRecord(raw), rootSchema) }));
  const ordinary = fields.filter((field) => !isAdvancedField(field.schema));
  const advanced = fields.filter((field) => isAdvancedField(field.schema));
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // A required advanced field with no value must never hide behind a closed
  // section: reloaded settings, a changed schema or a cleared value can all
  // leave one empty, and the browser cannot point at a hidden invalid input.
  const advancedNeedsValue = advanced.some(({ name, schema: field }) =>
    required.includes(name) && (values[name] === undefined || values[name] === null) && (field.default === undefined || field.default === null));
  const unitText = (unit: SettingsUnit) => ({
    usd: t('plugins.unit.usd', '$'),
    minutes: t('plugins.unit.minutes', 'minutes'),
    hours: t('plugins.unit.hours', 'hours'),
  })[unit];
  const renderField = ({ name, schema: field, nullable }: SettingsField & { name: string }) => {
    const type = schemaKind(field);
    const unit = settingsUnit(field);
    const label = fieldLabel(name, field.title, unit);
    const fieldId = `${id}-${name}`;
    const helpId = `${fieldId}-help`;
    const unitId = `${fieldId}-unit`;
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
    const formatNumber = (amount: number) => unit === 'usd' ? formatUsdAmount(amount) : String(amount);
    const placeholder = (value === undefined || value === null) && hasDefault ? (typeof fallback === 'number' ? formatNumber(fallback) : String(fallback))
      : nullable && (value === undefined || value === null) ? t('plugins.optionalValue', 'Optional') : undefined;
    // Returning to the default is offered only when it changes something.
    const resettable = !isRequired && value !== undefined && !(value === null && !hasDefault);
    // The unit is part of what the control means, so assistive technology hears it with the help text.
    const describedBy = [description ? helpId : null, unit ? unitId : null].filter(Boolean).join(' ') || undefined;
    const bounds = {
      min: typeof field.minimum === 'number' ? field.minimum : undefined, max: typeof field.maximum === 'number' ? field.maximum : undefined,
    };
    // Dollars step by the cent, unless a stored amount or the minimum (the
    // step's base) is finer: a cent step would then mark the saved value
    // invalid and block saving any other setting in the form.
    const centStep = [value, bounds.min].every((amount) => typeof amount !== 'number' || isWholeCents(amount));
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
        : unit === 'usd'
          ? <FormattedNumberInput id={fieldId} aria-describedby={describedBy} value={value} format={formatUsdAmount} disabled={disabled}
            required={isRequired} step={type === 'integer' ? 1 : centStep ? 0.01 : 'any'} placeholder={placeholder} {...bounds}
            onNumber={(next) => update(next ?? cleared)} />
          : <Input id={fieldId} aria-describedby={describedBy} type={type === 'string' ? 'text' : 'number'} value={typeof value === 'string' || typeof value === 'number' ? value : ''} disabled={disabled}
            required={isRequired} step={type === 'integer' ? 1 : 'any'} placeholder={placeholder} {...bounds}
            minLength={typeof field.minLength === 'number' ? field.minLength : undefined} maxLength={typeof field.maxLength === 'number' ? field.maxLength : undefined}
            onChange={(event) => update(type === 'string'
              ? (event.target.value === '' && nullable ? null : event.target.value)
              : Number.isFinite(event.target.valueAsNumber) ? event.target.valueAsNumber : cleared)} />;
    const withUnit = unit === null ? control
      : <Affixed>
        {unit === 'usd' ? <UnitText id={unitId}>{unitText(unit)}</UnitText> : null}
        {control}
        {unit !== 'usd' ? <UnitText id={unitId}>{unitText(unit)}</UnitText> : null}
      </Affixed>;
    return <Field key={name}>
      <LabelRow><label htmlFor={fieldId}>{label}</label>{isRequired ? <Required>{t('plugins.required', 'Required')}</Required> : null}</LabelRow>
      {description ? <Help id={helpId}>{description}</Help> : null}
      <Control>
        {withUnit}
        {resettable ? <Button type="button" variant="outline" size="sm" disabled={disabled} onClick={() => update(undefined)}>{hasDefault ? t('plugins.useDefault', 'Use default') : t('plugins.omit', 'Leave unset')}</Button> : null}
      </Control>
    </Field>;
  };
  return <Fields>
    {ordinary.map(renderField)}
    {advanced.length > 0 ? <Advanced open={advancedOpen || advancedNeedsValue} onToggle={(event) => {
      const section = event.currentTarget;
      // React leaves an unchanged `open` prop alone, so closing the section
      // while a required field is empty is undone here rather than by a render.
      if (advancedNeedsValue && !section.open) section.open = true;
      else setAdvancedOpen(section.open);
    }}>
      <summary>{t('plugins.advanced', 'Advanced')}</summary>
      <Fields>{advanced.map(renderField)}</Fields>
    </Advanced> : null}
  </Fields>;
}
