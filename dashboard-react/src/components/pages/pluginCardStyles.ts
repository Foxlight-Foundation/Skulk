import styled from 'styled-components';

/**
 * Shared parts of the plugin drawer's cards: the runtime card and each
 * capability node's card use the same header, section rhythm and notices.
 */

/** A bordered card in the plugin drawer. */
export const PluginCard = styled.article`
  display: flex; flex-direction: column; min-width: 0; border: 1px solid ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.lg}; background: ${({ theme }) => theme.colors.surface}; overflow-wrap: anywhere;
`;
/** The card's title row: title and secondary line, with a status pill at the end. */
export const CardHeader = styled.header`
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; padding: 18px 20px 16px;
  h3 { margin: 0; font-size: 17px; font-weight: 600; letter-spacing: -.01em; color: ${({ theme }) => theme.colors.text}; }
`;
/** Muted secondary text under a card title. */
export const Meta = styled.p`
  margin: 4px 0 0; font-size: 12.5px; color: ${({ theme }) => theme.colors.metadataText};
  code { font: 12px ${({ theme }) => theme.fonts.mono}; }
`;
/** One section of a card, with an uppercase label as its `h4`. */
export const Section = styled.section`
  display: flex; flex-direction: column; gap: 12px; padding: 18px 20px; border-top: 1px solid ${({ theme }) => theme.colors.border}; min-width: 0;
  h4 { margin: 0; font: 600 11px ${({ theme }) => theme.fonts.mono}; letter-spacing: .14em; text-transform: uppercase; color: ${({ theme }) => theme.colors.subtleText}; }
`;
/** A section's label row, with an optional control at the end. */
export const SectionHead = styled.div`display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;`;
/** One row of actions. */
export const ActionRow = styled.div`display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding-top: 4px;`;
/** Muted explanatory text. */
export const Muted = styled.p`margin: 0; font-size: 13px; line-height: 1.5; color: ${({ theme }) => theme.colors.textSecondary};`;
/** A notice, outlined; a problem is tinted. */
export const Notice = styled.p<{ $problem?: boolean }>`
  margin: 0; padding: 10px 12px; border-radius: ${({ theme }) => theme.radii.md}; font-size: 13px; line-height: 1.5;
  border: 1px solid ${({ theme, $problem }) => $problem ? theme.colors.borderDanger : theme.colors.border};
  background: ${({ theme, $problem }) => $problem ? theme.colors.errorBg : theme.colors.surfaceSunken};
  color: ${({ theme }) => theme.colors.text};
`;
