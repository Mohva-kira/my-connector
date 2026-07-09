import { create } from "zustand";

import type {
  EmailEnvelopeModel,
  SyncStage,
  SyncStats,
} from "../sync-experience/types";

const INITIAL_STATS: SyncStats = {
  emailsProcessed: 0,
  emailsTotal: 0,
  actionsFound: 0,
  deadlinesFound: 0,
  urgentFound: 0,
  progress: 0,
};

/** Maximum number of envelopes rendered on the conveyor at once (see architecture note: HUD carries true totals, the conveyor only shows a representative sample). */
export const MAX_VISIBLE_ENVELOPES = 30;

export interface SyncState {
  readonly stage: SyncStage;
  readonly stats: SyncStats;
  readonly currentTask: string | null;
  readonly envelopes: readonly EmailEnvelopeModel[];
}

export interface SyncActions {
  setStage: (stage: SyncStage) => void;
  setStats: (stats: Partial<SyncStats>) => void;
  setCurrentTask: (task: string | null) => void;
  upsertEnvelope: (envelope: EmailEnvelopeModel) => void;
  removeEnvelope: (id: string) => void;
  reset: () => void;
}

const INITIAL_STATE: SyncState = {
  stage: "CONNECTING",
  stats: INITIAL_STATS,
  currentTask: null,
  envelopes: [],
};

/**
 * Single source of truth for the sync/loading experience. Plain state +
 * setters only — this store never touches the EventBus directly. Diffing
 * and event emission is the responsibility of `useSyncGame`, keeping
 * "what changed" (this store) separate from "what one-off effect that
 * change should trigger" (the hook).
 */
export const useSyncStore = create<SyncState & SyncActions>((set) => ({
  ...INITIAL_STATE,

  setStage: (stage) => set({ stage }),

  setStats: (stats) =>
    set((state) => ({ stats: { ...state.stats, ...stats } })),

  setCurrentTask: (currentTask) => set({ currentTask }),

  upsertEnvelope: (envelope) =>
    set((state) => {
      const existingIndex = state.envelopes.findIndex(
        (e) => e.id === envelope.id,
      );
      if (existingIndex === -1) {
        const next = [...state.envelopes, envelope];
        return {
          envelopes:
            next.length > MAX_VISIBLE_ENVELOPES
              ? next.slice(next.length - MAX_VISIBLE_ENVELOPES)
              : next,
        };
      }
      const next = [...state.envelopes];
      next[existingIndex] = envelope;
      return { envelopes: next };
    }),

  removeEnvelope: (id) =>
    set((state) => ({
      envelopes: state.envelopes.filter((e) => e.id !== id),
    })),

  reset: () => set(INITIAL_STATE),
}));
