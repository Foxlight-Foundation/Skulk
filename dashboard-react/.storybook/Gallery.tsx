/* eslint-disable react-refresh/only-export-components -- Styled exports are React specimen components. */
import type { ReactNode } from 'react';
import styled from 'styled-components';

/** Shared specimen layout: production controls stay responsible for their own styling. */
export const Gallery = styled.main`
  padding: 48px 56px 80px; display: flex; flex-direction: column; gap: 56px;
  min-height: 100vh; background: ${({ theme }) => theme.colors.bg};
  color: ${({ theme }) => theme.colors.body};
  @media (max-width: 600px) { padding: 24px 16px 48px; gap: 32px; }
`;
/** Reference-sized specimen grid that stacks without shrinking controls on narrow screens. */
export const SpecimenGrid = styled.div`
  display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 340px), 1fr));
  gap: 12px; align-items: start;
`;
/** Bordered ridge surface around a labelled component specimen. */
export const Specimen = styled.div`
  min-width: 0; padding: 18px; border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: 12px; background: ${({ theme }) => theme.colors.surface};
  display: flex; flex-direction: column; gap: 14px;
`;
/** Wrapping row for controls, chips and compact states. */
export const SpecimenRow = styled.div`
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px; min-width: 0;
`;
/** Source and state caption, matching the reference's machine-label typography. */
export const SpecimenLabel = styled.div`
  font: 600 10px ${({ theme }) => theme.fonts.mono}; letter-spacing: .16em;
  text-transform: uppercase; color: ${({ theme }) => theme.colors.textMuted};
  overflow-wrap: anywhere;
`;
const Section = styled.section`
  display: flex; flex-direction: column; gap: 18px; min-width: 0;
  > h2 { margin: 0; font-size: 20px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }
`;
/** Named reference section, also used as a stable screenshot target. */
export function GallerySection({ title, children }: { title: string; children: ReactNode }) {
  return <Section aria-label={title}><h2>{title}</h2>{children}</Section>;
}
const Header = styled.header`
  display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; flex-wrap: wrap;
  h1 { margin: 6px 0 0; font-size: 34px; font-weight: 700; letter-spacing: -.025em; color: ${({ theme }) => theme.colors.text}; }
  p { margin: 8px 0 0; max-width: 640px; font-size: 14px; line-height: 1.55; color: ${({ theme }) => theme.colors.textSecondary}; }
  code { font: 11px ${({ theme }) => theme.fonts.mono}; color: ${({ theme }) => theme.colors.textMuted}; border: 1px solid ${({ theme }) => theme.colors.border}; border-radius: 6px; padding: 4px 8px; }
`;
/** Gallery heading and typography key; fixture copy never represents a live cluster. */
export function GalleryHeader({ title, children }: { title: string; children: ReactNode }) {
  return <Header><div><SpecimenLabel>Foxlight · Skulk dashboard</SpecimenLabel><h1>{title}</h1><p>{children}</p></div>
    <SpecimenRow><code>Instrument Sans</code><code>JetBrains Mono</code><code>4px grid</code><code>radius 4 · 8 · 12 · 16</code></SpecimenRow>
  </Header>;
}
