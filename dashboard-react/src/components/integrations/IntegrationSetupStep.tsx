import type { ReactNode } from 'react';
import styled from 'styled-components';

const Step = styled.section`display: grid; gap: 10px; min-width: 0;`;
const Heading = styled.div`display: flex; align-items: center; flex-wrap: wrap; gap: 8px; min-width: 0; h3 { flex: 1; min-width: 0; margin: 0; font-size: 14px; font-weight: 600; color: ${({ theme }) => theme.colors.text}; }`;
const Number = styled.span`flex-shrink: 0; width: 20px; height: 20px; display: grid; place-items: center; border-radius: 50%; background: ${({ theme }) => theme.colors.selected}; color: ${({ theme }) => theme.colors.text}; font: 600 11px ${({ theme }) => theme.fonts.mono};`;
const Actions = styled.div`margin-left: auto; min-width: 0; max-width: 100%; @media (max-width: 480px) { flex-basis: 100%; display: flex; justify-content: flex-end; }`;

/** Numbered setup instruction; completion must come from evidence owned by the caller. */
export function IntegrationSetupStep({ number, title, actions, children }: {
  number: number;
  title: string;
  /** Optional recipe controls aligned with the step heading. */
  actions?: ReactNode;
  children: ReactNode;
}) {
  return <Step><Heading><Number aria-hidden="true">{number}</Number><h3>{title}</h3>{actions && <Actions>{actions}</Actions>}</Heading>{children}</Step>;
}
