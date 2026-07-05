import styled from 'styled-components';
import { FiSettings, FiDatabase, FiMessageSquare, FiSun, FiMoon } from 'react-icons/fi';
import { MdHub } from 'react-icons/md';
import { VscBug } from 'react-icons/vsc';
import { useAppDispatch, useAppSelector } from '../../store/hooks';
import { uiActions } from '../../store/slices/uiSlice';
import { useSkulkTranslation } from '../../i18n/tolgee';
import type { NavRoute } from './HeaderNav';

/**
 * Props for the phone-width navigation sheet that replaces the header's
 * inline nav links and icon buttons below the mobile breakpoint.
 */
export interface MobileMenuSheetProps {
  /** Whether the sheet is expanded. The component stays mounted so the slide animation can run both ways. */
  open: boolean;
  /** The currently active route, highlighted in the menu. */
  activeRoute: NavRoute;
  /** Navigate to a route. The sheet closes itself via `onClose` after calling this. */
  onNavigate: (route: NavRoute) => void;
  /** Open the settings dialog. */
  onOpenSettings: () => void;
  /** Close the sheet (backdrop tap, or after any menu action). */
  onClose: () => void;
}

const Backdrop = styled.div<{ $open: boolean }>`
  position: fixed;
  inset: 0;
  z-index: 18;
  background: ${({ theme }) => theme.colors.shadowStrong};
  opacity: ${({ $open }) => ($open ? 1 : 0)};
  visibility: ${({ $open }) => ($open ? 'visible' : 'hidden')};
  transition: opacity 0.2s ease, visibility 0.2s;
`;

const Sheet = styled.nav<{ $open: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 19; /* under the header (20) so the sheet emerges from beneath it */
  padding: 76px 12px 12px; /* clears the header; items start just below it */
  background: ${({ theme }) => theme.colors.header};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  box-shadow: 0 12px 32px ${({ theme }) => theme.colors.shadowStrong};
  display: flex;
  flex-direction: column;
  gap: 4px;
  transform: translateY(${({ $open }) => ($open ? '0' : '-100%')});
  transition: transform 0.25s ease;
`;

const MenuRow = styled.button<{ $active?: boolean }>`
  all: unset;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: ${({ theme }) => theme.radii.md};
  font-size: ${({ theme }) => theme.fontSizes.nav};
  font-family: ${({ theme }) => theme.fonts.body};
  color: ${({ $active, theme }) => ($active ? theme.colors.gold : theme.colors.text)};
  background: ${({ $active, theme }) => ($active ? theme.colors.goldBg : 'transparent')};
  border: 1px solid ${({ $active, theme }) => ($active ? theme.colors.goldDim : 'transparent')};

  &:active {
    background: ${({ theme }) => theme.colors.goldBg};
  }
`;

const Divider = styled.div`
  height: 1px;
  margin: 6px 4px;
  background: ${({ theme }) => theme.colors.border};
`;

/**
 * Full-width sheet that slides down from under the header at phone width,
 * carrying everything the desktop header shows inline: the three routes,
 * the observability panel toggle, settings, and the theme switch. Every
 * action closes the sheet so the content behind it is immediately visible.
 */
export function MobileMenuSheet({ open, activeRoute, onNavigate, onOpenSettings, onClose }: MobileMenuSheetProps) {
  const { t } = useSkulkTranslation();
  const dispatch = useAppDispatch();
  const themeName = useAppSelector((s) => s.ui.theme);
  const observabilityPanelOpen = useAppSelector((s) => s.ui.observabilityPanelOpen);

  const go = (route: NavRoute) => {
    onNavigate(route);
    onClose();
  };

  return (
    <>
      <Backdrop $open={open} onClick={onClose} aria-hidden />
      <Sheet $open={open} aria-hidden={!open} aria-label={t('header.mobileMenu', 'Menu')}>
        <MenuRow $active={activeRoute === 'cluster'} onClick={() => go('cluster')} tabIndex={open ? 0 : -1}>
          <MdHub size={18} /> {t('header.nav.cluster', 'Cluster')}
        </MenuRow>
        <MenuRow $active={activeRoute === 'model-store'} onClick={() => go('model-store')} tabIndex={open ? 0 : -1}>
          <FiDatabase size={18} /> {t('header.nav.modelStore', 'Model Store')}
        </MenuRow>
        <MenuRow $active={activeRoute === 'chat'} onClick={() => go('chat')} tabIndex={open ? 0 : -1}>
          <FiMessageSquare size={18} /> {t('header.nav.chat', 'Chat')}
        </MenuRow>
        <Divider />
        <MenuRow
          $active={observabilityPanelOpen}
          onClick={() => {
            dispatch(observabilityPanelOpen ? uiActions.closeObservability() : uiActions.openObservability({}));
            onClose();
          }}
          tabIndex={open ? 0 : -1}
        >
          <VscBug size={18} /> {t('header.observability', 'Observability')}
        </MenuRow>
        <MenuRow
          onClick={() => {
            onOpenSettings();
            onClose();
          }}
          tabIndex={open ? 0 : -1}
        >
          <FiSettings size={18} /> {t('header.settings', 'Settings')}
        </MenuRow>
        <MenuRow onClick={() => dispatch(uiActions.toggleTheme())} tabIndex={open ? 0 : -1}>
          {themeName === 'dark' ? <FiSun size={18} /> : <FiMoon size={18} />}
          {themeName === 'dark'
            ? t('header.switchToLightTheme', 'Switch to light theme')
            : t('header.switchToDarkTheme', 'Switch to dark theme')}
        </MenuRow>
      </Sheet>
    </>
  );
}
