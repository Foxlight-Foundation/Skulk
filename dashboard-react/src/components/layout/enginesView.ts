import type { EnginePackage } from '../../types/engines';
import type { Translate } from './remotePairingMessages';

/** One build of an engine on a machine's disk, and what it is to the machine. */
export interface EngineBuildRow {
  build: string;
  /**
   * `inUse`: new launches use it. `unusable`: the build the machine holds and
   * cannot use. `earlier`: the build that was in use before, which the machine
   * keeps. `other`: on disk and none of these.
   */
  role: 'inUse' | 'unusable' | 'earlier' | 'other';
}

/** First characters of a build identity, enough to tell two builds apart. */
export function shortBuild(build: string): string {
  return build.slice(0, 12);
}

/**
 * List the builds a machine has on disk for one engine: the one it reports
 * first, the earlier one next, the rest in a stable order.
 *
 * An offered engine has nothing on disk, so it has no rows.
 */
export function engineBuildRows(enginePackage: EnginePackage): EngineBuildRow[] {
  if (enginePackage.status === 'offered') return [];
  const reported = enginePackage.build ?? null;
  const earlier = enginePackage.rollbackBuild ?? null;
  const onDisk = new Set(enginePackage.installedBuilds ?? []);
  if (reported) onDisk.add(reported);
  const role = (build: string): EngineBuildRow['role'] => {
    if (build === reported) return enginePackage.status === 'active' ? 'inUse' : 'unusable';
    return build === earlier ? 'earlier' : 'other';
  };
  const rank = { inUse: 0, unusable: 0, earlier: 1, other: 2 } as const;
  return [...onDisk]
    .map(build => ({ build, role: role(build) }))
    .sort((left, right) => rank[left.role] - rank[right.role] || left.build.localeCompare(right.build));
}

/** Name the kind of hardware a build runs on, in words a person recognizes. */
export function laneLabel(lane: string, t: Translate): string {
  switch (lane) {
    case 'metal': return t('engines.lane.metal', 'Apple GPU');
    case 'cuda': return t('engines.lane.cuda', 'NVIDIA GPU');
    case 'rocm': return t('engines.lane.rocm', 'AMD GPU (ROCm)');
    case 'vulkan': return t('engines.lane.vulkan', 'GPU (Vulkan)');
    case 'cpu': return t('engines.lane.cpu', 'Processor only');
    default: return lane;
  }
}

/** Say why a machine cannot use the build it holds. */
export function engineProblemSentence(problem: string | null | undefined, t: Translate): string {
  switch (problem) {
    case 'installation_damaged':
      return t('engines.problem.damaged', 'Its files on this machine are not intact. Use another build, or remove the engine so the machine installs it again.');
    case 'skulk_version_unsupported':
    case 'protocol_unsupported':
    case 'adapter_python_unsupported':
      return t('engines.problem.version', 'This version of Skulk cannot run it.');
    case 'packages_disabled':
      return t('engines.problem.disabled', 'Add-on engines are turned off on this machine.');
    case null:
    case undefined:
    case '':
      return t('engines.problem.unknown', 'The machine cannot use it and did not say why. Its log has the reason.');
    default:
      return t('engines.problem.code', 'The machine cannot use it ({code}).', { code: problem });
  }
}
