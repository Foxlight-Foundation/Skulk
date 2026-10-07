import type { InstanceStatus, NodeRunnerState } from '../components/cluster/RunningInstanceCard';
import type { SkulkTranslate } from '../i18n/tolgee';
import { engineInstallSize } from './engineInstall';

/** Runner status → instance status mapping for the running-instance cards. */

/** Collapse a single runner's tagged status into the per-node category the
 *  instance card renders. The runner lifecycle is idle -> connecting ->
 *  connected -> loading -> loaded -> warming up -> ready; everything that is not
 *  a terminal ready/failed/stopping state reads as "loading", except the initial
 *  RunnerIdle which reads as "pending" (spawned, not yet driven). */
export function runnerNodeState(runner: Record<string, unknown> | undefined): NodeRunnerState {
  if (!runner) return 'pending';
  const key = Object.keys(runner)[0];
  if (key === 'RunnerReady' || key === 'RunnerRunning') return 'ready';
  if (key === 'RunnerFailed') return 'failed';
  if (key === 'RunnerShuttingDown' || key === 'RunnerShutdown') return 'stopping';
  if (key === 'RunnerIdle') return 'pending';
  return 'loading';
}

/**
 * Derive one running-instance card's status and message from its runners.
 *
 * @param runnerIds - The instance's runner ids.
 * @param runners - Every runner's tagged status from cluster state.
 * @param t - Translation function.
 * @param engineInstallBytes - The approximate download of the on-demand engine
 *   (the video engine) the instance's node is installing before the model can
 *   load, zero when its size is unknown, or null when no install is running.
 * @returns The status category, an optional message, and load progress.
 */
export function deriveInstanceStatus(
  runnerIds: string[],
  runners: Record<string, Record<string, unknown>>,
  t: SkulkTranslate,
  engineInstallBytes: number | null = null,
): { status: InstanceStatus; message?: string; progress?: number } {
  if (runnerIds.length === 0) {
    return {
      status: 'loading',
      message: t('app.instanceStatus.waitingForRunners', 'Waiting for runners...'),
    };
  }

  const statuses = runnerIds.map((rid) => runners[rid]);

  // If any runner has failed, the instance is failed
  const failed = statuses.find((s) => s && 'RunnerFailed' in s);
  if (failed) {
    const inner = failed.RunnerFailed as Record<string, unknown> | undefined;
    return { status: 'failed', message: inner?.errorMessage as string | undefined };
  }

  // If any runner is shutting down
  if (statuses.some((s) => s && ('RunnerShuttingDown' in s || 'RunnerShutdown' in s))) {
    return { status: 'shutting_down' };
  }

  // If all runners are ready or running
  const allReady = statuses.every((s) => s && ('RunnerReady' in s || 'RunnerRunning' in s));
  if (allReady) {
    const anyRunning = statuses.some((s) => s && 'RunnerRunning' in s);
    return { status: anyRunning ? 'running' : 'ready' };
  }

  // If any runner is warming up
  if (statuses.some((s) => s && 'RunnerWarmingUp' in s)) {
    return { status: 'warming_up' };
  }

  // Loading — try to extract progress from RunnerLoading
  const loading = statuses.find((s) => s && 'RunnerLoading' in s);
  if (loading) {
    const inner = loading.RunnerLoading as Record<string, unknown> | undefined;
    const loaded = inner?.layersLoaded as number | undefined;
    const total = inner?.totalLayers as number | undefined;
    const progress = loaded != null && total != null && total > 0
      ? Math.round((loaded / total) * 100)
      : undefined;
    const message = loaded != null && total != null
      ? t('app.instanceStatus.downloadingLayers', 'Downloading layers {loaded}/{total}...', { loaded, total })
      : t('app.instanceStatus.loadingModel', 'Loading model...');
    return { status: 'loading', message, progress };
  }

  // The runner waits idle while its node installs an on-demand engine (the
  // video engine) beside the model download.
  if (engineInstallBytes !== null) {
    const size = engineInstallSize(engineInstallBytes);
    return {
      status: 'loading',
      message: size
        ? t('app.instanceStatus.installingVideoEngineSize', 'Installing video engine (about {size})...', { size })
        : t('app.instanceStatus.installingVideoEngine', 'Installing video engine...'),
    };
  }

  // Connecting or idle
  return { status: 'loading', message: t('app.instanceStatus.connecting', 'Connecting...') };
}
