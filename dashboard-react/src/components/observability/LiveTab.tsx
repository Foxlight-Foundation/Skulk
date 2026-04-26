import styled from 'styled-components';

/**
 * Live tab placeholder. Phase 3 (#120) wires this to:
 *
 * - Cluster health header (hang-rate counter, tracing toggle, master id, connectivity)
 * - Cross-rank flight-recorder timeline (`/v1/diagnostics/cluster/timeline`)
 * - Live event stream tail (`/events`)
 *
 * Phase 1 ships the panel scaffold only; the Live tab is intentionally an empty
 * affordance so the navigation pattern is testable end-to-end before the heavy
 * visualization work lands.
 */

const Wrap = styled.div`
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding: 24px 12px;
`;

const Heading = styled.h3`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.md};
  color: ${({ theme }) => theme.colors.text};
`;

const Body = styled.p`
  margin: 0;
  font-family: ${({ theme }) => theme.fonts.body};
  font-size: ${({ theme }) => theme.fontSizes.sm};
  line-height: 1.55;
  color: ${({ theme }) => theme.colors.textSecondary};
`;

const Coming = styled.div`
  margin-top: 8px;
  padding: 12px 14px;
  border: 1px dashed ${({ theme }) => theme.colors.border};
  border-radius: ${({ theme }) => theme.radii.md};
  font-family: ${({ theme }) => theme.fonts.mono};
  font-size: ${({ theme }) => theme.fontSizes.xs};
  color: ${({ theme }) => theme.colors.textMuted};
`;

export function LiveTab() {
  return (
    <Wrap>
      <Heading>Live cluster health</Heading>
      <Body>
        This view will show the cross-rank flight-recorder timeline, hang-rate counter,
        cluster-wide tracing toggle, and live event stream tail. Right now it's the empty
        scaffold delivered in Phase 1 of the observability consolidation.
      </Body>
      <Coming>
        Coming in Phase 3 (issue #120). Underlying data is already exposed at{' '}
        <code>/v1/diagnostics/cluster/timeline</code>; this tab just hasn't rendered it yet.
      </Coming>
    </Wrap>
  );
}
