import { useState } from 'react';
import type { Meta, StoryObj } from '@storybook/react-vite';
import { HeaderNav } from './HeaderNav';

const meta: Meta<typeof HeaderNav> = {
  title: 'Layout/HeaderNav',
  component: HeaderNav,
  parameters: { layout: 'fullscreen' },
  decorators: [(Story) => <div style={{ background: '#000' }}><Story /></div>],
};

export default meta;
type Story = StoryObj<typeof HeaderNav>;

export const Default: Story = {
  args: {
    showHome: true,
    onHome: () => {},
    showSidebarToggle: true,
    sidebarVisible: true,
  },
};

export const WithDownloadProgress: Story = {
  args: {
    showHome: true,
    showSidebarToggle: true,
    sidebarVisible: true,
    downloadProgress: { count: 3, percentage: 67 },
  },
};

export const Interactive: Story = {
  render: () => {
    const [sidebar, setSidebar] = useState(true);
    const [mobile, setMobile] = useState(false);
    const [right, setRight] = useState(false);
    return (
      <HeaderNav
        showHome
        onHome={() => alert('Home')}
        showSidebarToggle
        sidebarVisible={sidebar}
        onToggleSidebar={() => setSidebar(!sidebar)}
        showMobileMenuToggle
        mobileMenuOpen={mobile}
        onToggleMobileMenu={() => setMobile(!mobile)}
        mobileRightOpen={right}
        onToggleMobileRight={() => setRight(!right)}
        downloadProgress={{ count: 2, percentage: 45 }}
      />
    );
  },
};

export const Minimal: Story = {
  args: { showHome: false },
  name: 'Minimal (no toggles)',
};
