import React, { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { type ActionOut } from "../actionsApi";
import AppShell from "../components/AppShell";
import DiscoverProjectsModal from "../DiscoverProjectsModal";
import PortfolioAssistant from "../PortfolioAssistant";
import { apiFetch } from "../apiClient";
import { parseApiError, parseResponseJson } from "../apiUtils";
import type { Me } from "../SaasPanels";

type BriefResponse = {
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

function greetingName(email: string): string {
  const local = email.split("@")[0] ?? email;
  // .find(Boolean) plutôt que [0] : ignore les segments vides produits par
  // des séparateurs en tête/doublés (ex. "__jean.dupont@…" → "jean", pas "").
  const first = local.split(/[.\-_+]/).find(Boolean) ?? local;
  return first ? first.charAt(0).toUpperCase() + first.slice(1) : email;
}

function formatSince(since: string | null): string {
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

function formatDeadline(iso: string): string {
  try {
    return new Intl.DateTimeFormat("fr-FR", { day: "2-digit", month: "short" }).format(new Date(iso));
  } catch {
    return iso;
  }
}

type StatTone = "info" | "warning" | "danger";

function BriefStat({
  label,
  value,
  tone,
  loading,
}: {
  label: string;
  value: number | undefined;
  tone: StatTone;
  loading: boolean;
}) {
  const toneClass: Record<StatTone, string> = {
    info: "text-info",
    warning: "text-warning",
    danger: "text-danger",
  };
  return (
    <div className="rounded-2xl border border-border-default bg-surface p-4 shadow-sm">
      <p className={`text-2xl font-semibold ${toneClass[tone]}`}>{loading ? "…" : value ?? 0}</p>
      <p className="mt-1 text-xs text-text-secondary">{label}</p>
    </div>
  );
}

export default function BriefPage({ me }: { me: Me }) {
  const [brief, setBrief] = useState<BriefResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showDiscoverModal, setShowDiscoverModal] = useState(false);
  const [discoverSuccess, setDiscoverSuccess] = useState<string | null>(null);

  const loadBrief = useCallback(async () => {
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
  }, []);

  useEffect(() => {
    void loadBrief();
  }, [loadBrief]);

  function handleProjectsProcessed(created: number, updated: number) {
    setShowDiscoverModal(false);
    const parts: string[] = [];
    if (created > 0) parts.push(`${created} projet${created > 1 ? "s" : ""} ajouté${created > 1 ? "s" : ""}`);
    if (updated > 0) parts.push(`${updated} projet${updated > 1 ? "s" : ""} mis à jour`);
    setDiscoverSuccess(parts.length > 0 ? `${parts.join(", ")}.` : null);
    void loadBrief();
  }

  const counts = brief?.counts;

  return (
    <AppShell me={me}>
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text-primary">
            Bonjour {greetingName(me.email)}
          </h1>
          <p className="mt-1 text-sm text-text-muted">
            {brief ? (
              <>Voici ce qui a changé depuis {formatSince(brief.since)}.</>
            ) : (
              "Je prépare votre point du jour…"
            )}
          </p>
        </div>
        <button
          type="button"
          onClick={() => {
            setDiscoverSuccess(null);
            setShowDiscoverModal(true);
          }}
          className="rounded-xl bg-accent-primary px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-accent-hover"
        >
          Analyser mes emails
        </button>
      </header>

      {error ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          {error}
        </div>
      ) : null}

      {discoverSuccess ? (
        <div
          role="status"
          className="mb-6 rounded-2xl border border-success/30 bg-success/10 px-4 py-3 text-sm text-success"
        >
          {discoverSuccess}{" "}
          <Link to="/projects" className="font-medium underline">
            Voir mes projets
          </Link>
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <BriefStat label="Nouveaux projets" value={counts?.new_projects} tone="info" loading={loading} />
            <BriefStat
              label="Actions en attente"
              value={counts?.pending_actions}
              tone="warning"
              loading={loading}
            />
            <BriefStat
              label="Échéances cette semaine"
              value={counts?.upcoming_deadlines}
              tone="warning"
              loading={loading}
            />
            <BriefStat
              label="Échanges importants"
              value={counts?.important_emails}
              tone="info"
              loading={loading}
            />
            <BriefStat
              label="Projets à risque"
              value={counts?.at_risk_projects}
              tone="danger"
              loading={loading}
            />
          </div>

          <div className="rounded-2xl border border-border-default bg-surface p-5 shadow-sm">
            <h2 className="text-sm font-semibold text-text-primary">
              Ce que je vous recommande aujourd&apos;hui
            </h2>
            {loading ? (
              <p className="mt-3 text-sm text-text-muted">Chargement…</p>
            ) : brief && brief.recommended_actions.length > 0 ? (
              <ul className="mt-3 space-y-2">
                {brief.recommended_actions.map((a) => (
                  <li
                    key={a.id}
                    className="rounded-xl border border-border-subtle bg-bg-tertiary/40 px-3 py-2.5"
                  >
                    <p className="text-sm text-text-primary">{a.description}</p>
                    <p className="mt-0.5 text-xs text-text-muted">
                      {a.project_name}
                      {a.deadline ? ` · échéance ${formatDeadline(a.deadline)}` : ""}
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mt-3 text-sm text-text-muted">
                Rien ne nécessite votre attention pour le moment — tout est sous contrôle.
              </p>
            )}
            <Link
              to="/actions"
              className="mt-3 inline-block text-xs font-medium text-accent-primary hover:underline"
            >
              Voir toutes les actions →
            </Link>
          </div>
        </div>

        <PortfolioAssistant />
      </div>

      {showDiscoverModal ? (
        <DiscoverProjectsModal
          onClose={() => setShowDiscoverModal(false)}
          onProjectsProcessed={handleProjectsProcessed}
        />
      ) : null}
    </AppShell>
  );
}
