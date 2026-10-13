/** How long a setup run waits for its operation to appear after an unconfirmed reply. */
export const UNCONFIRMED_WAIT_MS = 30_000;

/** How often a run whose reply was not confirmed keeps looking for its operation. */
export const UNCONFIRMED_POLL_MS = 5_000;
