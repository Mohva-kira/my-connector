import React, { useEffect, useState } from "react";
import { apiFetch } from "./apiClient";

type OAuthConnection = {
  id: string;
  provider: string;
  email: string;
  scopes: string | null;
  created_at: string;
  updated_at: string;
};

const PROVIDER_LABELS: Record<string, string> = {
  gmail: "Gmail",
  outlook: "Outlook",
};

async function parseJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text.trim()) return {};
  return JSON.parse(text) as unknown;
}

function errorDetail(data: unknown, fallback: string): string {
  return data && typeof data === "object" && "detail" in data
    ? String((data as { detail: unknown }).detail)
    : fallback;
}

/** Section "Comptes connectés" de SettingsPage : liste, connecte (Gmail/Outlook
 * via redirection OAuth) et déconnecte des `OAuthConnection`. Provider-agnostic
 * côté API (`/api/oauth/connections`) — n'ajoute rien de spécifique à un
 * provider ici, juste les libellés d'affichage. */
export default function ConnectedAccounts() {
  const [connections, setConnections] = useState<OAuthConnection[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busyProvider, setBusyProvider] = useState<string | null>(null);
  const [callbackBanner, setCallbackBanner] = useState<{ ok: boolean; provider: string } | null>(null);

  async function loadConnections() {
    try {
      const res = await apiFetch("/api/oauth/connections");
      const data = await parseJson(res);
      if (!res.ok) throw new Error(errorDetail(data, "Impossible de charger les comptes connectés"));
      setConnections(data as OAuthConnection[]);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Erreur");
    }
  }

  useEffect(() => {
    void loadConnections();
  }, []);

  // Lit une seule fois le retour de callback OAuth (?oauth=success|error&provider=...)
  // émis par api/routers/oauth.py, affiche un statut, puis nettoie l'URL pour
  // ne pas le réafficher à un rafraîchissement de page.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const oauth = params.get("oauth");
    if (!oauth) return;
    const provider = params.get("provider") ?? "compte";
    setCallbackBanner({ ok: oauth === "success", provider });
    params.delete("oauth");
    params.delete("provider");
    const rest = params.toString();
    window.history.replaceState({}, "", `${window.location.pathname}${rest ? `?${rest}` : ""}`);
    void loadConnections();
  }, []);

  async function connect(provider: string) {
    setErr(null);
    setBusyProvider(provider);
    try {
      const res = await apiFetch(`/api/oauth/${provider}/authorize`);
      const data = await parseJson(res);
      if (!res.ok) throw new Error(errorDetail(data, "Échec de la connexion"));
      const url = (data as { url?: string }).url;
      if (!url) throw new Error("URL d'autorisation manquante");
      window.location.href = url;
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Erreur");
      setBusyProvider(null);
    }
  }

  async function disconnect(id: string) {
    setErr(null);
    try {
      const res = await apiFetch(`/api/oauth/connections/${id}`, { method: "DELETE" });
      if (!res.ok && res.status !== 204) {
        const data = await parseJson(res);
        throw new Error(errorDetail(data, "Échec de la déconnexion"));
      }
      await loadConnections();
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Erreur");
    }
  }

  const connectedProviders = new Set((connections ?? []).map((c) => c.provider));

  return (
    <section className="rounded-2xl border border-border-default bg-surface p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-text-primary">Comptes connectés</h2>
      <p className="mt-1 text-sm text-text-muted">
        Gmail ou Outlook, en plus (ou à la place) de la boîte IMAP ci-dessus.
      </p>

      {callbackBanner ? (
        <div
          role="status"
          className={`mt-4 rounded-xl border px-3 py-2 text-sm ${
            callbackBanner.ok
              ? "border-success/30 bg-success/10 text-success"
              : "border-danger/30 bg-danger/10 text-danger"
          }`}
        >
          {callbackBanner.ok
            ? `${PROVIDER_LABELS[callbackBanner.provider] ?? callbackBanner.provider} connecté avec succès.`
            : `Échec de la connexion à ${PROVIDER_LABELS[callbackBanner.provider] ?? callbackBanner.provider}.`}
        </div>
      ) : null}

      {err ? (
        <div role="alert" className="mt-4 rounded-xl border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          {err}
        </div>
      ) : null}

      <ul className="mt-4 space-y-2">
        {(connections ?? []).map((c) => (
          <li
            key={c.id}
            className="flex items-center justify-between rounded-xl border border-border-default bg-bg-tertiary/40 px-3 py-2 text-sm"
          >
            <div>
              <span className="font-medium text-text-primary">{PROVIDER_LABELS[c.provider] ?? c.provider}</span>
              <span className="ml-2 text-text-muted">{c.email}</span>
            </div>
            <button
              type="button"
              className="text-xs font-medium text-danger underline decoration-danger/40 underline-offset-2 hover:text-danger"
              onClick={() => void disconnect(c.id)}
            >
              Déconnecter
            </button>
          </li>
        ))}
        {connections !== null && connections.length === 0 ? (
          <li className="text-sm text-text-muted">Aucun compte connecté pour l&apos;instant.</li>
        ) : null}
      </ul>

      <div className="mt-4 flex flex-wrap gap-2">
        {(["gmail", "outlook"] as const).map((provider) => (
          <button
            key={provider}
            type="button"
            disabled={busyProvider === provider}
            className="rounded-full border border-border-default bg-bg-tertiary px-4 py-2 text-sm font-medium text-text-primary shadow-sm hover:opacity-90 disabled:opacity-50"
            onClick={() => void connect(provider)}
          >
            {busyProvider === provider
              ? "Redirection…"
              : connectedProviders.has(provider)
                ? `Connecter un autre compte ${PROVIDER_LABELS[provider]}`
                : `Connecter ${PROVIDER_LABELS[provider]}`}
          </button>
        ))}
      </div>
    </section>
  );
}
