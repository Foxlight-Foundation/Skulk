import styled, { css, useTheme } from 'styled-components';
import { FiSettings, FiMenu, FiX, FiSidebar, FiDatabase, FiMessageSquare, FiSun, FiMoon } from 'react-icons/fi';
import { MdHub } from 'react-icons/md';
import { VscBug } from 'react-icons/vsc';
import { Button } from '../common/Button';
import SkulkIcon from '../icons/SkulkIcon';
import type { Theme } from '../../theme';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';

export type NavRoute = 'cluster' | 'model-store' | 'chat';

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
  instanceCount?: number;
  instancesHealthy?: boolean;
  downloadProgress?: { count: number; percentage: number } | null;
  warnings?: { level: 'error' | 'warning'; items: { level: 'error' | 'warning'; message: string }[] } | null;
  onOpenSettings?: () => void;
  className?: string;
}

/* ---- styles ---- */

const Nav = styled.header`
  z-index: 20;
  background: ${({ theme }) => theme.colors.header};
  border-bottom: none;
  background-image: ${({ theme }) => theme.colors.headerBorder};
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
      color: ${({ theme }) => theme.colors.gold};
      border-color: ${({ theme }) => theme.colors.goldDim};
    `}
`;

const LogoBtn = styled.button<{ $disabled: boolean }>`
  all: unset;
  cursor: ${({ $disabled }) => ($disabled ? 'default' : 'pointer')};
  display: flex;
  align-items: center;
  gap: 8px;
  transition: opacity 0.15s;

  &:hover {
    opacity: ${({ $disabled }) => ($disabled ? 1 : 0.85)};
  }
`;

const LogoText = styled.span`
  font-size: ${({ theme }) => theme.fontSizes.xxl};
  font-weight: 700;
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.text};
  filter: drop-shadow(0 0 4px ${({ theme }) => theme.colors.border});
`;

const VersionTag = styled.sup`
  font-size: 10px;
  font-weight: 400;
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ theme }) => theme.colors.textSecondary};
  margin-left: 2px;
  position: relative;
  top: -4px;
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
  font-family: ${({ theme }) => theme.fonts.body};
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

const IconToggle = styled.span<{ $active: boolean }>`
  cursor: pointer;
  display: flex;
  align-items: center;
  color: ${({ $active, theme }) => $active ? theme.colors.gold : theme.colors.textMuted};
  transition: color 0.15s;

  &:hover {
    color: ${({ theme }) => theme.colors.text};
  }
`;

const WarningDot = styled.div<{ $level: 'error' | 'warning' }>`
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: help;

  &:hover > .warning-tooltip {
    opacity: 1;
    visibility: visible;
  }
`;

const WarningCircle = styled.div<{ $level: 'error' | 'warning' }>`
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: ${({ $level, theme}) => $level === 'error' ? theme.colors.error : theme.colors.warning};
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 700;
  font-family: ${({ theme }) => theme.fonts.body};
  /* Always white — sits on a saturated red/amber circle in both palettes. */
  color: #ffffff;
`;

const WarningTooltip = styled.div`
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  padding-top: 8px;
  width: 300px;
  opacity: 0;
  visibility: hidden;
  transition: opacity 0.2s, visibility 0.2s;
  z-index: 50;
`;

const WarningTooltipInner = styled.div`
  background: ${({ theme }) => theme.colors.surfaceElevated};
  border: 1px solid ${({ theme }) => theme.colors.goldDim};
  border-radius: ${({ theme }) => theme.radii.md};
  padding: 10px 12px;
  backdrop-filter: blur(8px);
  box-shadow: 0 8px 32px ${({ theme }) => theme.colors.shadowStrong};
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const WarningItem = styled.div<{ $level: 'error' | 'warning' }>`
  font-size: ${({ theme }) => theme.fontSizes.sm};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ $level, theme}) => $level === 'error' ? theme.colors.errorText : theme.colors.warningText};
  line-height: 1.4;
  display: flex;
  align-items: flex-start;
  gap: 6px;

  &::before {
    content: '●';
    color: ${({ $level, theme}) => $level === 'error' ? theme.colors.error : theme.colors.warning};
    flex-shrink: 0;
  }
`;

const InstanceToggle = styled.button<{ $healthy: boolean; $active: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2px;
  border-radius: 50%;
  transition: all 0.15s;

  &:hover {
    filter: brightness(1.2);
  }
`;

/* ---- icons (react-icons) ---- */

const MenuIcon = () => <FiMenu size={18} />;
const CloseIcon = () => <FiX size={18} />;
const SidebarIcon = () => <FiSidebar size={18} />;
const ClusterIcon = () => <MdHub size={16} />;
const StoreIcon = () => <FiDatabase size={16} />;
const ChatIcon = () => <FiMessageSquare size={16} />;
const ObservabilityIcon = () => <VscBug size={16} />;
const SettingsIcon = () => <FiSettings size={16} />;

function ProgressCircle({ count, percentage }: { count: number; percentage: number }) {
  const theme = useTheme() as Theme;
  const r = 10;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - percentage / 100);

  return (
    <DownloadBadge>
      <svg width="28" height="28" viewBox="0 0 28 28">
        <circle cx="14" cy="14" r={r} fill="none" stroke={theme.colors.borderStrong} strokeWidth="2" />
        <circle
          cx="14" cy="14" r={r}
          fill="none" stroke={theme.colors.gold} strokeWidth="2"
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          transform="rotate(-90 14 14)"
          style={{ transition: 'stroke-dashoffset 0.3s ease-out' }}
        />
        <text x="14" y="14" textAnchor="middle" dominantBaseline="central" fill={theme.colors.gold} fontSize="8" fontFamily="monospace">
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
  activeRoute = 'cluster',
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
  instanceCount = 0,
  instancesHealthy = true,
  downloadProgress = null,
  warnings = null,
  onOpenSettings,
  className,
}: HeaderNavProps) {
  const theme = useTheme() as Theme;
  const dispatch = useAppDispatch();
  const themeName = useAppSelector((s) => s.ui.theme);
  const toggleTheme = () => dispatch(uiActions.toggleTheme());
  // Observability panel is global UI state — the button toggles it open/closed and
  // visually reflects whether it's currently visible. Distinct from `activeRoute`
  // because the panel overlays the current route rather than navigating away.
  const observabilityPanelOpen = useAppSelector((s) => s.ui.observabilityPanelOpen);
  const openObservability = (tab?: 'live' | 'node' | 'traces', nodeId?: string) =>
    dispatch(uiActions.openObservability({ tab, nodeId }));
  const closeObservability = () => dispatch(uiActions.closeObservability());
  const navigate = (route: NavRoute) => {
    onNavigate?.(route);
    if (route === 'cluster') onHome?.();
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
          <IconToggle $active={sidebarVisible} onClick={onToggleSidebar} aria-label="Toggle sidebar">
            <SidebarIcon />
          </IconToggle>
        )}
        <LogoBtn $disabled={!showHome} onClick={showHome ? () => navigate('cluster') : undefined}>
          <SkulkIcon size={32} color={theme.colors.text} />
          <LogoText>Skulk<VersionTag>{__APP_VERSION__}</VersionTag></LogoText>
        </LogoBtn>
        {warnings && warnings.items.length > 0 && (
          <WarningDot $level={warnings.level}>
            <WarningCircle $level={warnings.level}>!</WarningCircle>
            <WarningTooltip className="warning-tooltip">
              <WarningTooltipInner>
                {warnings.items.map((item, i) => (
                  <WarningItem key={i} $level={item.level}>{item.message}</WarningItem>
                ))}
              </WarningTooltipInner>
            </WarningTooltip>
          </WarningDot>
        )}
      </LeftGroup>

      <RightGroup>
        {downloadProgress && <ProgressCircle count={downloadProgress.count} percentage={downloadProgress.percentage} />}

        <NavLink $active={activeRoute === 'cluster'} onClick={() => navigate('cluster')}>
          <ClusterIcon /> Cluster
        </NavLink>

        <NavLink $active={activeRoute === 'model-store'} onClick={() => navigate('model-store')}>
          <StoreIcon /> Model Store
        </NavLink>

        <NavLink $active={activeRoute === 'chat'} onClick={() => navigate('chat')}>
          <ChatIcon /> Chat
        </NavLink>

        {instanceCount > 0 && (
          <InstanceToggle
            $healthy={instancesHealthy}
            $active={mobileRightOpen}
            onClick={onToggleMobileRight}
            aria-label="Toggle instances panel"
            aria-pressed={mobileRightOpen}
          >
            <svg width="28" height="28" viewBox="0 0 28 28">
              <circle
                cx="14" cy="14" r="11"
                fill="none"
                stroke={instancesHealthy ? theme.colors.healthy : theme.colors.error}
                strokeWidth="2"
                opacity={mobileRightOpen ? 1 : 0.7}
              />
              <text
                x="14" y="14"
                textAnchor="middle"
                dominantBaseline="central"
                fill={instancesHealthy ? theme.colors.healthy : theme.colors.error}
                fontSize="13"
                fontFamily="'Outfit', sans-serif"
                fontWeight="700"
              >
                {instanceCount}
              </text>
            </svg>
          </InstanceToggle>
        )}

        <Button
          variant="ghost"
          size="lg"
          icon
          onClick={toggleTheme}
          aria-label={themeName === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          aria-pressed={themeName === 'light'}
          title={themeName === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        >
          {themeName === 'dark' ? <FiSun size={16} /> : <FiMoon size={16} />}
        </Button>

        <Button
          variant={observabilityPanelOpen ? 'outline' : 'ghost'}
          size="lg"
          icon
          onClick={() => (observabilityPanelOpen ? closeObservability() : openObservability())}
          aria-label="Observability"
          aria-pressed={observabilityPanelOpen}
          title="Observability"
        >
          <ObservabilityIcon />
        </Button>

        <Button variant="ghost" size="lg" icon onClick={() => onOpenSettings?.()} aria-label="Settings">
          <SettingsIcon />
        </Button>
      </RightGroup>
    </Nav>
  );
}
