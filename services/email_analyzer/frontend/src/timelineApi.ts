import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de TimelineEventOut (api/routers/brief.py::get_timeline) — Timeline
// unifiée (pivot exécutif, Phase 12). Premier module frontend à consommer
// /api/timeline.
export type TimelineEvent = {
  type: "email" | "summary_update" | "action_suggested" | "meeting";
  project_id: string;
  project_name: string;
  at: string;
  label: string;
};

export async function fetchTimeline(limit = 50): Promise<TimelineEvent[]> {
  const res = await apiFetch(`/api/timeline?limit=${limit}`);
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as TimelineEvent[];
}
