import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "../apiClient";
import { parseApiError, parseResponseJson } from "../apiUtils";
import type { ActionOut } from "../actionsApi";

export type BriefResponse = {
  since: string | null;
  counts: {
    new_projects: number;
    pending_actions: number;
    upcoming_deadlines: number;
    important_emails: number;
    at_risk_projects: number;
  };
  recommended_actions: ActionOut[];
};

export function greetingName(email: string): string {
  const local = email.split("@")[0] ?? email;
  // .find(Boolean) plutôt que [0] : ignore les segments vides produits par
  // des séparateurs en tête/doublés (ex. "__jean.dupont@…" → "jean", pas "").
  const first = local.split(/[.\-_+]/).find(Boolean) ?? local;
  return first ? first.charAt(0).toUpperCase() + first.slice(1) : email;
}

export function formatSince(since: string | null): string {
  if (!since) return "votre première visite";
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      day: "2-digit",
      month: "long",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(since));
  } catch {
    return since;
  }
}

// Porte GET /api/brief (ancien BriefPage.tsx::loadBrief) — alimente la
// salutation et les cartes d'action rapide SaaS de ChatHome. `enabled=false`
// (mode legacy) n'appelle jamais /api/brief (503 sans SaaS, voir
// api/main.py::assert_saas_enabled).
export function useBrief(enabled: boolean) {
  const [brief, setBrief] = useState<BriefResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    if (!enabled) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiFetch("/api/brief");
      const data = await parseResponseJson(res);
      if (!res.ok) throw new Error(parseApiError(data));
      setBrief(data as BriefResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setLoading(false);
    }
  }, [enabled]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { brief, loading, error, reload };
}
