import { useCallback, useEffect, useRef, useState } from 'react';
import styled from 'styled-components';
import { Field, type FieldSize } from './Field';

export interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  size?: FieldSize;
  /** If set, onChange fires after the user stops typing for this many ms. */
  debounceMs?: number;
  autoFocus?: boolean;
  className?: string;
}

const SearchIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);

const ClearButton = styled.button`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  font-size: ${({ theme }) => theme.fontSizes.label};
  color: ${({ theme }) => theme.colors.textMuted};
  transition: color 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
  }
`;

export function SearchBar({
  value,
  onChange,
  placeholder = 'Search…',
  size = 'md',
  debounceMs,
  autoFocus,
  className,
}: SearchBarProps) {
  const [local, setLocal] = useState(value);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  // Sync external value changes
  useEffect(() => {
    setLocal(value);
  }, [value]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const v = e.target.value;
      setLocal(v);
      if (debounceMs) {
        clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => onChange(v), debounceMs);
      } else {
        onChange(v);
      }
    },
    [onChange, debounceMs],
  );

  const handleClear = useCallback(() => {
    setLocal('');
    onChange('');
    clearTimeout(timerRef.current);
  }, [onChange]);

  // Cleanup timer on unmount
  useEffect(() => () => clearTimeout(timerRef.current), []);

  return (
    <Field
      value={local}
      onChange={handleChange}
      placeholder={placeholder}
      size={size}
      autoFocus={autoFocus}
      icon={<SearchIcon />}
      rightElement={
        local ? (
          <ClearButton onClick={handleClear} aria-label="Clear search">
            ✕
          </ClearButton>
        ) : undefined
      }
      className={className}
    />
  );
}
