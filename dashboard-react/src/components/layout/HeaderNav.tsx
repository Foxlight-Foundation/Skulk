import styled, { css } from 'styled-components';

export interface HeaderNavProps {
  showHome?: boolean;
  onHome?: () => void;
  showSidebarToggle?: boolean;
  sidebarVisible?: boolean;
  onToggleSidebar?: () => void;
  showMobileMenuToggle?: boolean;
  mobileMenuOpen?: boolean;
  onToggleMobileMenu?: () => void;
  showMobileRightToggle?: boolean;
  mobileRightOpen?: boolean;
  onToggleMobileRight?: () => void;
  downloadProgress?: { count: number; percentage: number } | null;
  className?: string;
}

/* ---- styles ---- */

const Nav = styled.header`
  z-index: 20;
  background: ${({ theme }) => theme.colors.surface};
  padding: 16px 24px 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
`;

const LeftGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const RightGroup = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ToggleBtn = styled.button<{ $active: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(179, 179, 179, 0.3);
  color: ${({ $active }) => ($active ? '#FFD700' : 'rgba(179,179,179,0.8)')};
  transition: all 0.15s;

  &:hover {
    border-color: rgba(255, 215, 0, 0.5);
    color: #FFD700;
  }
`;

const LogoBtn = styled.button<{ $disabled: boolean }>`
  all: unset;
  cursor: ${({ $disabled }) => ($disabled ? 'default' : 'pointer')};
  display: flex;
  align-items: center;
  transition: opacity 0.15s;

  &:hover {
    opacity: ${({ $disabled }) => ($disabled ? 1 : 0.85)};
  }
`;

const LogoText = styled.span`
  font-size: 24px;
  font-weight: 700;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: #FFD700;
  letter-spacing: 4px;
  filter: drop-shadow(0 0 4px rgba(255, 215, 0, 0.3));
`;

const NavLink = styled.a`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid rgba(179, 179, 179, 0.3);
  font-size: 11px;
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1px;
  color: rgba(179, 179, 179, 0.8);
  transition: all 0.15s;

  &:hover {
    border-color: rgba(255, 215, 0, 0.5);
    color: #FFD700;
  }
`;

const DownloadBadge = styled.div`
  position: relative;
  width: 28px;
  height: 28px;
`;

/* ---- icons ---- */

const MenuIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="18" x2="21" y2="18" />
  </svg>
);

const CloseIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M6 18L18 6M6 6l12 12" />
  </svg>
);

const SidebarIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <rect x="3" y="3" width="18" height="18" rx="2" /><line x1="9" y1="3" x2="9" y2="21" />
  </svg>
);

const PanelIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <rect x="3" y="3" width="18" height="18" rx="2" /><line x1="15" y1="3" x2="15" y2="21" />
  </svg>
);

const DownloadIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
    <polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
  </svg>
);

const SettingsIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

function ProgressCircle({ count, percentage }: { count: number; percentage: number }) {
  const r = 10;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - percentage / 100);

  return (
    <DownloadBadge>
      <svg width="28" height="28" viewBox="0 0 28 28">
        <circle cx="14" cy="14" r={r} fill="none" stroke="rgba(179,179,179,0.2)" strokeWidth="2" />
        <circle
          cx="14" cy="14" r={r}
          fill="none" stroke="#FFD700" strokeWidth="2"
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 14 14)"
          style={{ transition: 'stroke-dashoffset 0.3s ease-out' }}
        />
        <text x="14" y="14" textAnchor="middle" dominantBaseline="central" fill="#FFD700" fontSize="8" fontFamily="monospace">
          {count}
        </text>
      </svg>
    </DownloadBadge>
  );
}

/* ---- component ---- */

export function HeaderNav({
  showHome = true,
  onHome,
  showSidebarToggle = false,
  sidebarVisible = true,
  onToggleSidebar,
  showMobileMenuToggle = false,
  mobileMenuOpen = false,
  onToggleMobileMenu,
  showMobileRightToggle = false,
  mobileRightOpen = false,
  onToggleMobileRight,
  downloadProgress = null,
  className,
}: HeaderNavProps) {
  return (
    <Nav className={className}>
      <LeftGroup>
        {showMobileMenuToggle && (
          <ToggleBtn $active={mobileMenuOpen} onClick={onToggleMobileMenu} aria-label="Toggle mobile menu" aria-pressed={mobileMenuOpen}>
            {mobileMenuOpen ? <CloseIcon /> : <MenuIcon />}
          </ToggleBtn>
        )}
        {showSidebarToggle && (
          <ToggleBtn $active={sidebarVisible} onClick={onToggleSidebar} aria-label="Toggle sidebar" aria-pressed={sidebarVisible}>
            <SidebarIcon />
          </ToggleBtn>
        )}
        <LogoBtn $disabled={!showHome} onClick={showHome ? onHome : undefined}>
          <LogoText>EXO</LogoText>
        </LogoBtn>
      </LeftGroup>

      <RightGroup>
        {downloadProgress && <ProgressCircle count={downloadProgress.count} percentage={downloadProgress.percentage} />}

        <NavLink href="/#/downloads">
          <DownloadIcon /> Downloads
        </NavLink>

        <NavLink href="/#/settings">
          <SettingsIcon /> Settings
        </NavLink>

        {showMobileRightToggle && (
          <ToggleBtn $active={mobileRightOpen} onClick={onToggleMobileRight} aria-label="Toggle right panel" aria-pressed={mobileRightOpen}>
            <PanelIcon />
          </ToggleBtn>
        )}
      </RightGroup>
    </Nav>
  );
}
