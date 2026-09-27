import { describe, expect, it } from 'vitest';
import { dashboardUrlOn, isTailnetAddress, pluginsPath, tailnetAddress } from './hostDashboard';

describe('isTailnetAddress', () => {
  it('accepts only Tailscale-assigned addresses', () => {
    expect(isTailnetAddress('100.64.0.1')).toBe(true);
    expect(isTailnetAddress('100.127.255.254')).toBe(true);
    expect(isTailnetAddress('fd7a:115c:a1e0::1')).toBe(true);
    expect(isTailnetAddress('[fd7a:115c:a1e0:ab12::7]')).toBe(true);
    expect(isTailnetAddress('100.63.255.255')).toBe(false);
    expect(isTailnetAddress('100.128.0.1')).toBe(false);
    expect(isTailnetAddress('192.168.1.20')).toBe(false);
    expect(isTailnetAddress('100.64.0.300')).toBe(false);
    expect(isTailnetAddress('fe80::1')).toBe(false);
  });
});

describe('tailnetAddress', () => {
  it('prefers a node\'s Tailscale IPv4 over its IPv6 and ignores other networks', () => {
    const node = { network_interfaces: [
      { name: 'en0', addresses: ['192.168.1.20', 'fe80::1'] },
      { name: 'utun4', addresses: ['fd7a:115c:a1e0::7', '100.70.1.2/32'] },
    ] };
    expect(tailnetAddress(node)).toBe('100.70.1.2');
    expect(tailnetAddress({ network_interfaces: [{ addresses: ['fd7a:115c:a1e0::7'] }] })).toBe('fd7a:115c:a1e0::7');
    expect(tailnetAddress({ network_interfaces: [{ addresses: ['10.0.0.4'] }] })).toBeNull();
    expect(tailnetAddress(undefined)).toBeNull();
  });
});

describe('dashboardUrlOn', () => {
  it('addresses the node on this page\'s scheme and port', () => {
    expect(dashboardUrlOn('100.70.1.2', pluginsPath(), { protocol: 'http:', port: '52415' })).toBe('http://100.70.1.2:52415/plugins');
    expect(dashboardUrlOn('fd7a:115c:a1e0::7', pluginsPath('managed.ab'), { protocol: 'https:', port: '' })).toBe('https://[fd7a:115c:a1e0::7]/plugins?plugin=managed.ab');
  });
});
