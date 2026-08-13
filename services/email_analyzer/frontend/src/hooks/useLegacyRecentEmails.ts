import { useEffect, useState } from "react";
import { apiFetch } from "../apiClient";
import { parseApiError, parseResponseJson } from "../apiUtils";

export type RecentEmail = { subject: string; from: string; date: string };

const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 60_000;

// Alimente les bulles de suggestion du chat en mode legacy (pas de DB, donc
// pas de /api/timeline — voir ChatHome.tsx) : POST /api/recent-emails lance un
// job arq headers-only (analysis_tasks.py::run_recent_emails_legacy), poll
// générique sur /api/analyze/{job_id} jusqu'à done/error. Best-effort : une
// erreur ici ne doit jamais bloquer le chat, juste laisser la liste vide.
export function useLegacyRecentEmails(enabled: boolean): RecentEmail[] {
  const [emails, setEmails] = useState<RecentEmail[]>([]);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    async function run() {
      try {
        const startRes = await apiFetch("/api/recent-emails?days=14&limit=5", { method: "POST" });
        const startData = (await parseResponseJson(startRes)) as Record<string, unknown>;
        if (!startRes.ok) throw new Error(parseApiError(startData));
        const jobId = startData.job_id as string | undefined;
        if (!jobId) return;

        const deadline = Date.now() + POLL_TIMEOUT_MS;
        while (Date.now() < deadline && !cancelled) {
          await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
          const res = await apiFetch(`/api/analyze/${jobId}`);
          const data = (await parseResponseJson(res)) as Record<string, unknown>;
          if (!res.ok || cancelled) return;
          if (data.status === "done") {
            const result = data.result as { emails?: RecentEmail[] } | null;
            if (result?.emails) setEmails(result.emails);
            return;
          }
          if (data.status === "error") return;
        }
      } catch {
        // Suggestions best-effort : voir docstring ci-dessus.
      }
    }

    void run();
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  return emails;
}
