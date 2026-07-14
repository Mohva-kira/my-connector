import { apiFetch } from "./apiClient";
import { parseApiError, parseResponseJson } from "./apiUtils";

// Miroir de ActionOut (api/routers/actions.py) — partagé entre BriefPage,
// AgendaPage et ActionsPage pour éviter de redéclarer le type et les deux
// appels réseau à chaque page.
export type ActionOut = {
  id: string;
  project_id: string;
  project_name: string;
  description: string;
  deadline: string | null;
  status: string;
  created_at: string;
  // Détail (fiche "ouvrir une action") : pourquoi/qui/conseil — null pour les
  // actions créées avant cette colonne ou via le repli sans extraction
  // structurée (voir email_analyzer/analysis_tasks.py).
  rationale: string | null;
  stakeholder: string | null;
  advice: string | null;
};

export async function fetchActions(status?: string): Promise<ActionOut[]> {
  const qs = status ? `?status=${encodeURIComponent(status)}` : "";
  const res = await apiFetch(`/api/actions${qs}`);
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as ActionOut[];
}

export async function updateActionStatus(id: string, status: string): Promise<ActionOut> {
  const res = await apiFetch(`/api/actions/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  const data = await parseResponseJson(res);
  if (!res.ok) throw new Error(parseApiError(data));
  return data as ActionOut;
}
