import type { ReactNode } from 'react';
import styled from 'styled-components';

const Step = styled.section`display: grid; grid-template-columns: 20px minmax(0, 1fr); gap: 12px; min-width: 0;`;
const Number = styled.span`width: 20px; height: 20px; display: grid; place-items: center; border-radius: 50%; background: ${({ theme }) => theme.colors.selected}; color: ${({ theme }) => theme.colors.text}; font: 11px ${({ theme }) => theme.fonts.mono};`;
const Content = styled.div`min-width: 0; display: grid; gap: 14px; h3 { margin: 0; font-size: 14px; color: ${({ theme }) => theme.colors.text}; }`;

/** Numbered setup instruction; completion must come from evidence owned by the caller. */
export function IntegrationSetupStep({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return <Step><Number aria-hidden="true">{number}</Number><Content><h3>{title}</h3>{children}</Content></Step>;
}
