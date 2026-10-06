import type { NetworkInterfaceInfo } from '../types/topology';

/*
 * Links to another node's own dashboard over the tailnet.
 *
 * A node admits plugin management from a browser that reaches it over
 * Tailscale: the socket peer is a tailnet address the node's tailscaled
 * confirms, and the page's origin is that same address. So an owner on any
 * tailnet machine can manage a node's plugins by opening that node's own
 * dashboard. Nothing is relayed or proxied; the link only opens the page.
 */

const IPV4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/;

/** Whether an address is in Tailscale's assigned space: 100.64.0.0/10 or fd7a:115c:a1e0::/48. */
export function isTailnetAddress(address: string): boolean {
  const ipv4 = IPV4.exec(address.trim());
  if (ipv4) {
    const octets = ipv4.slice(1).map(Number);
    return octets.every((octet) => octet <= 255) && octets[0] === 100 && octets[1] >= 64 && octets[1] <= 127;
  }
  return address.trim().toLowerCase().replace(/^\[|\]$/g, '').startsWith('fd7a:115c:a1e0:');
}

/** The Tailscale address a node reports on its interfaces, IPv4 first; null when it reports none. */
export function tailnetAddress(node: { network_interfaces?: NetworkInterfaceInfo[] } | undefined | null): string | null {
  // An interface may report an address with its prefix length.
  const addresses = (node?.network_interfaces ?? []).flatMap((item) => item.addresses ?? []).map((address) => address.split('/')[0].trim());
  const tailnet = addresses.filter(isTailnetAddress);
  return tailnet.find((address) => IPV4.test(address)) ?? tailnet[0] ?? null;
}

/**
 * A dashboard path on the node reached at `address`, on this page's scheme and
 * port: every node serves its dashboard on the same port as the API.
 */
export function dashboardUrlOn(address: string, path: string, page: Pick<Location, 'protocol' | 'port'> = window.location): string {
  const bare = address.replace(/^\[|\]$/g, '');
  const host = bare.includes(':') ? `[${bare}]` : bare;
  const port = page.port ? `:${page.port}` : '';
  return `${page.protocol}//${host}${port}${path.startsWith('/') ? path : `/${path}`}`;
}

/** The Plugins page, opened on one installed plugin when one is named. */
export function pluginsPath(pluginId?: string | null): string {
  return pluginId ? `/plugins?plugin=${encodeURIComponent(pluginId)}` : '/plugins';
}
