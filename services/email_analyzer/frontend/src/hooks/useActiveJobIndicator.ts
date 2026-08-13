import { useEffect, useState } from "react";
import { apiFetch } from "../apiClient";
import { parseResponseJson } from "../apiUtils";

const POLL_INTERVAL_MS = 15_000;

// Indicateur global "analyse en cours" (badge AppShell) — simple détection
// de présence via GET /api/jobs/active, peu importe le déclencheur (bouton
// manuel, cron run_agenda_refresh/run_scheduled_sync, auto-sync au login).
// Volontairement beaucoup plus espacé que le polling de progression d'un
// job précis (useLegacyHomeChat.ts, 2.5s) : ce hook ne fait que savoir
// "y a-t-il quelque chose en cours ?", pas suivre un job donné.
export function useActiveJobIndicator(enabled: boolean) {
  const [active, setActive] = useState(false);

  useEffect(() => {
    if (!enabled) {
      setActive(false);
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const res = await apiFetch("/api/jobs/active");
        const data = await parseResponseJson(res);
        if (!cancelled && res.ok) {
          setActive(Boolean((data as { active?: boolean }).active));
        }
      } catch {
        // Silencieux : un indicateur de fond ne doit pas faire remonter
        // d'erreur visible, une panne réseau transitoire disparaîtra au
        // prochain tick.
      }
    }

    void poll();
    const id = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [enabled]);

  return active;
}
