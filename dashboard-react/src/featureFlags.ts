/**
 * Build-time dashboard feature switches. Each is a single `VITE_*` variable
 * read once so a feature can flip between default-on and opt-in without
 * touching the components that implement it.
 */

/**
 * Capability satellites in the topology: managed capability nodes drawn
 * around their host with a flyout of surfaces and actions. On unless the
 * build sets `VITE_CAPABILITY_SATELLITES=0`.
 */
export const CAPABILITY_SATELLITES_ENABLED = import.meta.env.VITE_CAPABILITY_SATELLITES !== '0';
