import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de MilestoneOut/ObjectiveOut (api/routers/objectives.py) —
// Objectives (pivot exécutif, Phase 11). Premier module frontend à
// consommer /api/objectives.
export type Milestone = {
  id: string;
  title: string;
  due_date: string | null;
  status: "pending" | "done";
};

export type Objective = {
  id: string;
  project_id: string | null;
  title: string;
  target_date: string | null;
  status: "on_track" | "at_risk" | "blocked" | "done";
  blocking_reason: string | null;
  milestones: Milestone[];
  completed_count: number;
  total_count: number;
  percent: number;
};

export async function fetchObjectives(projectId?: string): Promise<Objective[]> {
  const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
  const res = await apiFetch(`/api/objectives${qs}`);
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Objective[];
}

export async function createObjective(
  title: string,
  projectId?: string,
  targetDate?: string,
): Promise<Objective> {
  const res = await apiFetch("/api/objectives", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, project_id: projectId ?? null, target_date: targetDate ?? null }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Objective;
}

export async function updateObjectiveStatus(
  id: string,
  status: Objective["status"],
  blockingReason?: string,
): Promise<Objective> {
  const res = await apiFetch(`/api/objectives/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status, blocking_reason: blockingReason ?? null }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Objective;
}

export async function addMilestone(objectiveId: string, title: string): Promise<Objective> {
  const res = await apiFetch(`/api/objectives/${objectiveId}/milestones`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Objective;
}

export async function setMilestoneStatus(
  objectiveId: string,
  milestoneId: string,
  status: Milestone["status"],
): Promise<Objective> {
  const res = await apiFetch(`/api/objectives/${objectiveId}/milestones/${milestoneId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as Objective;
}
