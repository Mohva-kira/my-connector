import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de agenda.py (api/routers/agenda.py).
export type AppointmentOut = {
  id: string;
  project_id: string;
  project_name: string;
  description: string;
  scheduled_at: string;
  participants: string[];
  status: "tentative" | "confirmed" | "cancelled";
};

export type AgendaProjectOut = {
  project_id: string;
  project_name: string;
  sentiment: string | null;
  llm_risk_level: string | null;
  probable_next_contact_date: string | null;
  probable_next_contact_reason: string | null;
  probable_next_contact_confidence: string | null;
  last_processed_email_timestamp: string | null;
};

export type AgendaOut = {
  appointments: AppointmentOut[];
  awaiting_projects: AgendaProjectOut[];
  at_risk_projects: AgendaProjectOut[];
  agenda_updated_at: string | null;
};

export async function fetchAgenda(): Promise<AgendaOut> {
  const res = await apiFetch("/api/agenda");
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as AgendaOut;
}

export async function updateAppointmentStatus(
  id: string,
  status: "tentative" | "confirmed" | "cancelled",
): Promise<AppointmentOut> {
  const res = await apiFetch(`/api/agenda/appointments/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as AppointmentOut;
}

export async function requestAgendaRefresh(): Promise<{ job_id: string; status: string }> {
  const res = await apiFetch("/api/agenda/refresh", { method: "POST" });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as { job_id: string; status: string };
}

// Polling du job de rafraîchissement Agenda : même endpoint/cadence que le
// Fast-Track (ProjectHub.tsx::pollRefreshJob) — /api/analyze/{job_id} sert
// tous les types de jobs (voir jobs.py, générique par job_id). Timeout réduit
// (portée limitée aux projets en attente/en rouge d'un seul tenant).
const AGENDA_REFRESH_POLL_INTERVAL_MS = 2500;
const AGENDA_REFRESH_POLL_TIMEOUT_MS = 2 * 60 * 1000;

export async function pollAgendaRefreshJob(jobId: string): Promise<void> {
  const deadline = Date.now() + AGENDA_REFRESH_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, AGENDA_REFRESH_POLL_INTERVAL_MS));
    const res = await apiFetch(`/api/analyze/${jobId}`, {
      method: "GET",
      signal: AbortSignal.timeout(30_000),
    });
    const data = (await parseResponseJson(res)) as Record<string, unknown>;
    if (!res.ok) throw new Error(parseApiError(data));
    const status = data.status as string;
    if (status === "done") return;
    if (status === "error") throw new Error((data.error as string) || "Échec du rafraîchissement.");
  }
  throw new Error("Le rafraîchissement prend trop de temps. Réessayez plus tard.");
}
