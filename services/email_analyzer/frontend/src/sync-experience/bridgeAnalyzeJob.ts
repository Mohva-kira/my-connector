import type { SyncActions, SyncState } from "../store/syncStore";
import type { SyncStage } from "./types";

/** Shape of a GET /api/analyze/{job_id} tick, as returned by `api/main.py`. */
export interface AnalyzeJobTick {
  status: "pending" | "running" | "done" | "error";
  progress?: { processed: number; total: number } | null;
  partial?: Record<string, { nb_emails: number; emails_critiques: unknown[] }> | null;
}

type SyncStoreSlice = Pick<SyncState & SyncActions, "setStage" | "setStats" | "reset">;

/**
 * Maps a real analyze-job tick onto the sync store, using only the signals
 * the backend actually emits today (status + processed/total + per-project
 * emails_critiques counts — see Unit 3/6 in progress-tracker.md). The 6
 * intermediate SyncStage values (NORMALIZING..GENERATING_BRIEFING) are never
 * reached: the backend has no pipeline-stage signal yet, and this bridge
 * must not invent one.
 */
export function applyAnalyzeTickToStore(tick: AnalyzeJobTick, store: SyncStoreSlice): void {
  if (tick.status === "error") {
    store.reset();
    return;
  }

  const processed = tick.progress?.processed ?? 0;
  const total = tick.progress?.total ?? 0;
  const urgentFound = Object.values(tick.partial ?? {}).reduce(
    (sum, block) => sum + (block.emails_critiques?.length ?? 0),
    0,
  );

  // Stats must land before the stage flips to COMPLETED: `setStage` and
  // `setStats` are two separate store updates, each notifying subscribers
  // immediately (see useSyncGame's subscribe callback) — if COMPLETED were
  // set first, SYNC_COMPLETED would fire with the previous tick's stats.
  store.setStats({
    emailsProcessed: processed,
    emailsTotal: total,
    urgentFound,
    progress: total > 0 ? Math.round((processed / total) * 100) : 0,
  });

  const stage: SyncStage =
    tick.status === "done" ? "COMPLETED" : tick.status === "running" ? "IMPORTING" : "CONNECTING";
  store.setStage(stage);
}
