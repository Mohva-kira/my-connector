import { useEffect } from "react";

import { syncEventBus } from "../sync-experience/eventBus/EventBus";
import type { SyncStats } from "../sync-experience/types";
import { useSyncStore } from "../store/syncStore";

function emitStatDeltas(previous: SyncStats, current: SyncStats): void {
  const emailsDelta = current.emailsProcessed - previous.emailsProcessed;
  if (emailsDelta > 0) {
    syncEventBus.emit("EMAIL_IMPORTED", { count: emailsDelta });
  }

  const actionsDelta = current.actionsFound - previous.actionsFound;
  if (actionsDelta > 0) {
    syncEventBus.emit("ACTION_FOUND", { count: actionsDelta });
  }

  const deadlinesDelta = current.deadlinesFound - previous.deadlinesFound;
  if (deadlinesDelta > 0) {
    syncEventBus.emit("DEADLINE_FOUND", { count: deadlinesDelta });
  }

  const urgentDelta = current.urgentFound - previous.urgentFound;
  if (urgentDelta > 0) {
    syncEventBus.emit("URGENT_FOUND", { count: urgentDelta });
  }

  if (current.progress !== previous.progress) {
    syncEventBus.emit("PROGRESS_UPDATED", { progress: current.progress });
  }
}

/**
 * Subscribes the store to the EventBus: diffs each state transition against
 * the previous snapshot and emits the corresponding discrete events. Kept
 * free of React so it can be driven directly in a plain script to verify the
 * store -> EventBus flow without mounting a component tree.
 */
export function subscribeSyncStoreToEventBus(): () => void {
  return useSyncStore.subscribe((state, previousState) => {
    if (state.stage !== previousState.stage) {
      syncEventBus.emit("STAGE_CHANGED", {
        stage: state.stage,
        previousStage: previousState.stage,
      });
    }

    emitStatDeltas(previousState.stats, state.stats);

    if (state.stage === "COMPLETED" && previousState.stage !== "COMPLETED") {
      syncEventBus.emit("SYNC_COMPLETED", { stats: state.stats });
    }
  });
}

/**
 * The sole bridge between the Zustand store and the EventBus. Mount once
 * near the root of the sync experience. Presentational components never
 * read the store and the EventBus in the same place for the same fact —
 * aggregate numbers come from the store via selectors, one-off effects come
 * from the bus.
 */
export function useSyncGame(): void {
  useEffect(() => {
    return subscribeSyncStoreToEventBus();
  }, []);
}
