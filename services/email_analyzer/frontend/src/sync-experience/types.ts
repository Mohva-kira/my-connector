/**
 * Shared vocabulary for the Sync Experience module.
 * These types are the contract between the Zustand store, the EventBus,
 * and the presentational components — nothing here depends on React.
 */

/** Ordered pipeline stages, matching the backend processing pipeline 1:1. */
export type SyncStage =
  | "CONNECTING"
  | "IMPORTING"
  | "NORMALIZING"
  | "SCORING"
  | "AI_CLASSIFICATION"
  | "TASK_EXTRACTION"
  | "DEADLINE_DETECTION"
  | "GENERATING_BRIEFING"
  | "COMPLETED";

export const SYNC_STAGES: readonly SyncStage[] = [
  "CONNECTING",
  "IMPORTING",
  "NORMALIZING",
  "SCORING",
  "AI_CLASSIFICATION",
  "TASK_EXTRACTION",
  "DEADLINE_DETECTION",
  "GENERATING_BRIEFING",
  "COMPLETED",
];

/** Lifecycle of a single email envelope as it travels the visible conveyor. */
export type EmailVisualState =
  | "waiting"
  | "moving"
  | "processing"
  | "completed"
  | "archived"
  | "urgent";

export type EmailPriority = "low" | "medium" | "high" | "urgent";

export type EmailCategory = "client" | "personal" | "admin" | "spam";

/** A single envelope rendered on the conveyor (capped sample, not 1:1 with real volume). */
export interface EmailEnvelopeModel {
  readonly id: string;
  readonly priority: EmailPriority;
  readonly category: EmailCategory;
  readonly currentStation: SyncStage;
  readonly state: EmailVisualState;
}

/** Aggregate counters shown in the HUD — always reflect true backend totals. */
export interface SyncStats {
  readonly emailsProcessed: number;
  readonly emailsTotal: number;
  readonly actionsFound: number;
  readonly deadlinesFound: number;
  readonly urgentFound: number;
  readonly progress: number; // 0-100
}

/**
 * Typed payloads for every event the store can emit toward the UI layer.
 *
 * `count`-based events are derived by diffing aggregate counters in
 * `useSyncGame` (the hook only sees totals, not which email changed).
 * `emailId`-based events are emitted at the call site that mutates a
 * specific envelope, since only that call site knows which entity changed.
 */
export interface SyncEventMap {
  SYNC_STARTED: { emailsTotal: number };
  EMAIL_IMPORTED: { count: number };
  EMAIL_PROCESSED: { emailId: string; stage: SyncStage };
  ACTION_FOUND: { count: number };
  DEADLINE_FOUND: { count: number };
  URGENT_FOUND: { count: number };
  PROGRESS_UPDATED: { progress: number };
  STAGE_CHANGED: { stage: SyncStage; previousStage: SyncStage | null };
  SYNC_COMPLETED: { stats: SyncStats };
}

export type SyncEventName = keyof SyncEventMap;
