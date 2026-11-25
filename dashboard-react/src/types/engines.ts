/**
 * The cluster's add-on engines report (`GET /v1/engines`) and the answer to a
 * change on one machine (`POST /v1/engines/{engine_id}/...`).
 *
 * An add-on engine is an engine package: a program that runs models and is
 * installed on a machine separately from Skulk. Field meanings are documented
 * with the routes in the API guide.
 */

/** The accelerator class one build of an engine runs on. */
export type EngineLane = 'metal' | 'cuda' | 'rocm' | 'vulkan' | 'cpu';

/** What a person may ask a machine to do with an engine it holds. */
export type EngineAction = 'activate' | 'rollback' | 'remove';

/** How the last model load on one build ended on one machine. */
export interface EngineLoadResult {
  build: string;
  outcome: 'loaded' | 'refused' | 'failed';
  /** The engine's own refusal code; present only for `refused`. */
  code?: string | null;
  modelId: string;
  /** When the load ended, by the machine's own clock. */
  atUnixMs: number;
}

/** One engine on one accelerator class on one machine. */
export interface EnginePackage {
  engineId: string;
  lane: string;
  backendTag: string;
  /**
   * `active`: the machine holds a build and new launches use it.
   * `unusable`: it holds a build it cannot use; `problem` says why.
   * `offered`: it holds none and installs one when a model needs it.
   */
  status: 'active' | 'unusable' | 'offered';
  /** Identity of the active or offered build. */
  build?: string | null;
  engineVersion?: string | null;
  generation?: number | null;
  publisher?: string | null;
  /** How far an offered build is released. */
  channel?: 'stable' | 'candidate' | null;
  /** Every build of this engine on the machine's disk. */
  installedBuilds?: string[];
  /** The build that was in use before the active one, when it is kept. */
  rollbackBuild?: string | null;
  /** Short refusal code for an unusable build. */
  problem?: string | null;
  downloadBytes?: number | null;
  overheadBytes?: number | null;
  loadResults?: EngineLoadResult[];
}

/** One machine's entry for an engine. */
export interface EngineNodePackage {
  nodeId: string;
  package: EnginePackage;
}

/** One engine and every machine that holds it or is offered it. */
export interface EngineEntry {
  engineId: string;
  name: string;
  nodes: EngineNodePackage[];
}

/** The whole report. */
export interface EnginesReport {
  /** Whether add-on engines are turned on for the machine that answered. */
  packagesEnabled: boolean;
  engines: EngineEntry[];
}

/** A change to make to one engine on one machine. */
export interface EngineChangeRequest {
  engineId: string;
  action: EngineAction;
  nodeId: string;
  lane: string;
  /** The build to use or delete. Left out to delete the whole engine. */
  build?: string;
}

/** What the machine holds for the engine straight after a change. */
export interface EngineChange {
  nodeId: string;
  engineId: string;
  lane: string;
  action: EngineAction;
  /** Null when the machine no longer holds or is offered the engine. */
  package: EnginePackage | null;
}
