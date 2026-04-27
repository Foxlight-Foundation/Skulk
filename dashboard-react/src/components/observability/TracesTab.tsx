import styled from 'styled-components';
import { Button } from '../common/Button';
import { useUIStore } from '../../stores/uiStore';

/**
 * Traces tab placeholder. Phase 2 (#119) replaces this body with:
 *
 * - The trace list (currently lives in `TracesPage`)
 * - A custom Skulk-native waterfall renderer for the selected trace, replacing
 *   today's "open in Perfetto" popup which sends trace data to Google's hosted
 *   web app and routinely fails to popup blockers.
 *
 * For Phase 1 the tab keeps trace browsing reachable by sending the user back to
 * the existing `traces` route. The route itself is removed from the top nav (the
 * Observability button replaces the old `Traces` icon) but the route is still
 * registered, so the page renders normally when navigated to.
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

const Actions = styled.div`
  display: flex;
  gap: 8px;
  margin-top: 4px;
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

export function TracesTab() {
  const setActiveRoute = useUIStore((s) => s.setActiveRoute);
  const setSelectedTraceTaskId = useUIStore((s) => s.setSelectedTraceTaskId);
  const closeObservability = useUIStore((s) => s.closeObservability);

  const openLegacyTraces = () => {
    setSelectedTraceTaskId(null);
    setActiveRoute('traces');
    closeObservability();
  };

  return (
    <Wrap>
      <Heading>Saved traces</Heading>
      <Body>
        Inline trace browsing is coming with a Skulk-native waterfall renderer in
        Phase 2 — replacing the current Perfetto popup integration with something
        that doesn't ship trace data to a third-party web app and isn't fragile to
        popup blockers.
      </Body>
      <Actions>
        <Button variant="outline" size="sm" onClick={openLegacyTraces}>
          Open trace browser (legacy view)
        </Button>
      </Actions>
      <Coming>
        Coming in Phase 2 (issue #119).
      </Coming>
    </Wrap>
  );
}
