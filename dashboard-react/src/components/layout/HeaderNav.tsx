import styled, { css } from 'styled-components';
import { FiHome, FiDownload, FiSettings, FiMenu, FiX, FiSidebar } from 'react-icons/fi';
import { MdOutlineViewSidebar } from 'react-icons/md';
import { Button } from '../common/Button';

export type NavRoute = 'home' | 'downloads';

export interface HeaderNavProps {
  showHome?: boolean;
  onHome?: () => void;
  activeRoute?: NavRoute;
  onNavigate?: (route: NavRoute) => void;
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
  onOpenSettings?: () => void;
  className?: string;
}

/* ---- styles ---- */

const Nav = styled.header`
  z-index: 20;
  background: ${({ theme }) => theme.colors.surface};
  border-bottom: none;
  background-image: linear-gradient(to right, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.03));
  background-size: 100% 1px;
  background-position: bottom;
  background-repeat: no-repeat;
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

const ToggleBtn = styled(Button)<{ $active: boolean }>`
  ${({ $active }) =>
    $active &&
    css`
      color: #FFD700;
      border-color: rgba(255, 215, 0, 0.4);
    `}
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
  font-size: ${({ theme }) => theme.fontSizes.xxl};
  font-weight: 700;
  font-family: ${({ theme }) => theme.fonts.mono};
  color: #FFD700;
  filter: drop-shadow(0 0 4px rgba(255, 215, 0, 0.3));
`;

const NavLink = styled.button<{ $active?: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: ${({ theme }) => theme.radii.md};
  border: 1px solid ${({ $active, theme }) => $active ? theme.colors.goldDim : theme.colors.border};
  background: ${({ $active, theme }) => $active ? theme.colors.goldBg : 'transparent'};
  font-size: ${({ theme }) => theme.fontSizes.nav};
  font-family: ${({ theme }) => theme.fonts.mono};
  text-transform: uppercase;
  letter-spacing: 1px;
  color: ${({ $active, theme }) => $active ? theme.colors.gold : theme.colors.textSecondary};
  transition: all 0.15s;

  &:hover {
    border-color: ${({ theme }) => theme.colors.goldDim};
    color: ${({ theme }) => theme.colors.gold};
  }
`;

const DownloadBadge = styled.div`
  position: relative;
  width: 28px;
  height: 28px;
`;

/* ---- icons (react-icons) ---- */

const MenuIcon = () => <FiMenu size={18} />;
const CloseIcon = () => <FiX size={18} />;
const SidebarIcon = () => <FiSidebar size={18} />;
const PanelIcon = () => <MdOutlineViewSidebar size={18} />;
const DownloadIcon = () => <FiDownload size={16} />;
const HomeIcon = () => <FiHome size={16} />;
const SettingsIcon = () => <FiSettings size={16} />;

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
  activeRoute = 'home',
  onNavigate,
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
  onOpenSettings,
  className,
}: HeaderNavProps) {
  const navigate = (route: NavRoute) => {
    onNavigate?.(route);
    if (route === 'home') onHome?.();
  };

  return (
    <Nav className={className}>
      <LeftGroup>
        {showMobileMenuToggle && (
          <ToggleBtn variant="outline" size="lg" icon $active={mobileMenuOpen} onClick={onToggleMobileMenu} aria-label="Toggle mobile menu" aria-pressed={mobileMenuOpen}>
            {mobileMenuOpen ? <CloseIcon /> : <MenuIcon />}
          </ToggleBtn>
        )}
        {showSidebarToggle && (
          <ToggleBtn variant="outline" size="lg" icon $active={sidebarVisible} onClick={onToggleSidebar} aria-label="Toggle sidebar" aria-pressed={sidebarVisible}>
            <SidebarIcon />
          </ToggleBtn>
        )}
        <LogoBtn $disabled={!showHome} onClick={showHome ? () => navigate('home') : undefined}>
          <LogoText>Skulk</LogoText>
        </LogoBtn>
      </LeftGroup>

      <RightGroup>
        {downloadProgress && <ProgressCircle count={downloadProgress.count} percentage={downloadProgress.percentage} />}

        <NavLink $active={activeRoute === 'home'} onClick={() => navigate('home')}>
          <HomeIcon /> Home
        </NavLink>

        <NavLink $active={activeRoute === 'downloads'} onClick={() => navigate('downloads')}>
          <DownloadIcon /> Downloads
        </NavLink>

        <Button variant="ghost" size="lg" icon onClick={() => onOpenSettings?.()} aria-label="Settings">
          <SettingsIcon />
        </Button>

        {showMobileRightToggle && (
          <ToggleBtn variant="outline" size="lg" icon $active={mobileRightOpen} onClick={onToggleMobileRight} aria-label="Toggle right panel" aria-pressed={mobileRightOpen}>
            <PanelIcon />
          </ToggleBtn>
        )}
      </RightGroup>
    </Nav>
  );
}
