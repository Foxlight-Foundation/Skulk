/**
 * DirectoryBrowser
 *
 * Inline directory picker that fetches /filesystem/browse and lets the user
 * navigate down into subdirectories.  Mirrors the Svelte DirectoryBrowser component.
 */
import React, { useState, useCallback } from 'react';
import styled from 'styled-components';
import { browseFilesystem } from '../../api/client';
import type { DirectoryEntry } from '../../api/types';

interface Props {
  value: string;
  label?: string;
  onChange: (path: string) => void;
}

const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const Label = styled.label`
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
`;

const InputRow = styled.div`
  display: flex;
  gap: 8px;
`;

const TextInput = styled.input`
  flex: 1;
  background: ${({ theme }) => theme.colors.black};
  border: 1px solid ${({ theme }) => theme.colors.mediumGray};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 7px 10px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  color: ${({ theme }) => theme.colors.foreground};
  outline: none;

  &::placeholder { color: ${({ theme }) => theme.colors.lightGray}; opacity: 0.4; }
  &:focus { border-color: ${({ theme }) => theme.colors.yellow}; }
`;

const BrowseButton = styled.button`
  background: transparent;
  border: 1px solid ${({ theme }) => theme.colors.mediumGray};
  border-radius: ${({ theme }) => theme.radius.sm};
  padding: 7px 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  white-space: nowrap;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover {
    border-color: ${({ theme }) => theme.colors.yellow};
    color: ${({ theme }) => theme.colors.yellow};
  }
`;

const BrowserPanel = styled.div`
  background: ${({ theme }) => theme.colors.black};
  border: 1px solid ${({ theme }) => theme.colors.mediumGray};
  border-radius: ${({ theme }) => theme.radius.sm};
  overflow: hidden;
`;

const BrowserHeader = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-bottom: 1px solid ${({ theme }) => theme.colors.mediumGray};
  background: ${({ theme }) => `${theme.colors.darkGray}`};
`;

const CurrentPath = styled.span`
  flex: 1;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const HeaderButton = styled.button`
  background: transparent;
  border: none;
  padding: 2px 8px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 10px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.lightGray};
  cursor: pointer;
  border-radius: ${({ theme }) => theme.radius.sm};
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover { color: ${({ theme }) => theme.colors.yellow}; }
  &:disabled { opacity: 0.4; cursor: default; }
`;

const DirList = styled.ul`
  list-style: none;
  margin: 0;
  padding: 4px 0;
  max-height: 180px;
  overflow-y: auto;
`;

const DirItem = styled.li<{ $active?: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 12px;
  cursor: pointer;
  color: ${({ theme, $active }) => $active ? theme.colors.yellow : theme.colors.foreground};
  background: ${({ theme, $active }) => $active ? `${theme.colors.yellow}15` : 'transparent'};
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover {
    background: ${({ theme }) => `${theme.colors.mediumGray}`};
  }
`;

const SelectButton = styled.button`
  width: 100%;
  text-align: left;
  background: transparent;
  border: none;
  border-top: 1px solid ${({ theme }) => theme.colors.mediumGray};
  padding: 6px 12px;
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: ${({ theme }) => theme.colors.yellow};
  cursor: pointer;
  transition: ${({ theme }) => theme.transitions.fast};

  &:hover { background: ${({ theme }) => `${theme.colors.yellow}15`}; }
`;

const EmptyNote = styled.div`
  padding: 12px;
  font-size: 11px;
  color: ${({ theme }) => theme.colors.lightGray};
  text-align: center;
`;

const DirectoryBrowser: React.FC<Props> = ({ value, label, onChange }) => {
  const [open, setOpen] = useState(false);
  const [browsePath, setBrowsePath] = useState('/Volumes');
  const [dirs, setDirs] = useState<DirectoryEntry[]>([]);
  const [loading, setLoading] = useState(false);

  const browse = useCallback(async (path: string) => {
    setLoading(true);
    try {
      const result = await browseFilesystem(path);
      setBrowsePath(result.path);
      setDirs(result.directories);
    } catch {
      setDirs([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleOpen = () => {
    setOpen(true);
    void browse(value || '/Volumes');
  };

  const handleClose = () => setOpen(false);

  const handleUp = () => {
    const parent = browsePath.split('/').slice(0, -1).join('/') || '/';
    void browse(parent);
  };

  const handleSelect = () => {
    onChange(browsePath);
    setOpen(false);
  };

  return (
    <Wrapper>
      {label && <Label>{label}</Label>}
      <InputRow>
        <TextInput
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="/path/to/directory"
        />
        <BrowseButton type="button" onClick={open ? handleClose : handleOpen}>
          Browse
        </BrowseButton>
      </InputRow>
      {open && (
        <BrowserPanel>
          <BrowserHeader>
            <HeaderButton type="button" onClick={handleUp} disabled={browsePath === '/'}>
              ↑ Up
            </HeaderButton>
            <CurrentPath title={browsePath}>{browsePath}</CurrentPath>
            <HeaderButton type="button" onClick={handleClose}>✕</HeaderButton>
          </BrowserHeader>
          {loading ? (
            <EmptyNote>Loading…</EmptyNote>
          ) : dirs.length === 0 ? (
            <EmptyNote>No subdirectories</EmptyNote>
          ) : (
            <DirList>
              {dirs.map((d) => (
                <DirItem key={d.path} onClick={() => void browse(d.path)}>
                  📁 {d.name}
                </DirItem>
              ))}
            </DirList>
          )}
          <SelectButton type="button" onClick={handleSelect}>
            ✓ Select this directory
          </SelectButton>
        </BrowserPanel>
      )}
    </Wrapper>
  );
};

export default DirectoryBrowser;
