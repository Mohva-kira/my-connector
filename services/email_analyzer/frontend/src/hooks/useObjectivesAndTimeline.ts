import { useCallback, useEffect, useState } from "react";
import {
  addMilestone,
  createObjective,
  fetchObjectives,
  setMilestoneStatus,
  updateObjectiveStatus,
  type Objective,
} from "../objectivesApi";
import { fetchTimeline, type TimelineEvent } from "../timelineApi";

// Charge Objectives (Phase 11) + Timeline (Phase 12) pour la page
// /objectives — même pattern que useMemoryAndKnowledge (Unit 53) : ces deux
// piliers existent côté backend depuis Units 54-55 mais n'avaient jusqu'ici
// aucune UI. `enabled=false` (mode legacy) n'appelle jamais ces endpoints.
export function useObjectivesAndTimeline(enabled: boolean) {
  const [objectives, setObjectives] = useState<Objective[]>([]);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError(null);
    try {
      const [o, t] = await Promise.all([fetchObjectives(), fetchTimeline()]);
      setObjectives(o);
      setTimeline(t);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    void reload();
  }, [reload]);

  const addObjective = useCallback(async (title: string) => {
    const created = await createObjective(title);
    setObjectives((prev) => [created, ...prev]);
  }, []);

  const toggleObjectiveBlocked = useCallback(async (objective: Objective, reason?: string) => {
    const next = objective.status === "blocked" ? "on_track" : "blocked";
    const updated = await updateObjectiveStatus(objective.id, next, reason);
    setObjectives((prev) => prev.map((o) => (o.id === updated.id ? updated : o)));
  }, []);

  const addObjectiveMilestone = useCallback(async (objectiveId: string, title: string) => {
    const updated = await addMilestone(objectiveId, title);
    setObjectives((prev) => prev.map((o) => (o.id === updated.id ? updated : o)));
  }, []);

  const toggleMilestone = useCallback(
    async (objectiveId: string, milestoneId: string, currentStatus: "pending" | "done") => {
      const next = currentStatus === "done" ? "pending" : "done";
      const updated = await setMilestoneStatus(objectiveId, milestoneId, next);
      setObjectives((prev) => prev.map((o) => (o.id === updated.id ? updated : o)));
    },
    [],
  );

  return {
    objectives,
    timeline,
    loading,
    error,
    reload,
    addObjective,
    toggleObjectiveBlocked,
    addObjectiveMilestone,
    toggleMilestone,
  };
}
