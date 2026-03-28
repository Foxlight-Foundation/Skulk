import { useEffect, useRef, useState } from 'react';
import styled, { css, keyframes } from 'styled-components';
import { FiSettings, FiMenu, FiX, FiSidebar, FiDatabase, FiMessageSquare } from 'react-icons/fi';
import { MdHub } from 'react-icons/md';
import { MdOutlineViewSidebar } from 'react-icons/md';
import { Button } from '../common/Button';
import SkulkIcon from '../icons/SkulkIcon';

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
  onOpenSettings?: () => void;
  className?: string;
}

/* ---- styles ---- */

const Nav = styled.header`
  z-index: 20;
  background: ${({ theme }) => theme.colors.header};
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
  color: #ffffff;
  filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.2));
`;

const VersionTag = styled.sup`
  font-size: 10px;
  font-weight: 400;
  font-family: ${({ theme }) => theme.fonts.body};
  color: rgba(255, 255, 255, 0.88);
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
const PanelIcon = () => <MdOutlineViewSidebar size={18} />;
const ClusterIcon = () => <MdHub size={16} />;
const StoreIcon = () => <FiDatabase size={16} />;
const ChatIcon = () => <FiMessageSquare size={16} />;
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
  onOpenSettings,
  className,
}: HeaderNavProps) {
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
          <ToggleBtn variant="outline" size="lg" icon $active={sidebarVisible} onClick={onToggleSidebar} aria-label="Toggle sidebar" aria-pressed={sidebarVisible}>
            <SidebarIcon />
          </ToggleBtn>
        )}
        <LogoBtn $disabled={!showHome} onClick={showHome ? () => navigate('cluster') : undefined}>
          <SkulkIcon size={32} color="#ffffff" />
          <LogoText>Skulk<VersionTag>{__APP_VERSION__}</VersionTag></LogoText>
        </LogoBtn>
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
                stroke={instancesHealthy ? '#4ade80' : '#ef4444'}
                strokeWidth="2"
                opacity={mobileRightOpen ? 1 : 0.7}
              />
              <text
                x="14" y="14"
                textAnchor="middle"
                dominantBaseline="central"
                fill={instancesHealthy ? '#4ade80' : '#ef4444'}
                fontSize="13"
                fontFamily="'Outfit', sans-serif"
                fontWeight="700"
              >
                {instanceCount}
              </text>
            </svg>
          </InstanceToggle>
        )}

        <Button variant="ghost" size="lg" icon onClick={() => onOpenSettings?.()} aria-label="Settings">
          <SettingsIcon />
        </Button>
      </RightGroup>
    </Nav>
  );
}
