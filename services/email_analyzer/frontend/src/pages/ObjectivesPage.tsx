import { useState } from "react";
import { Navigate } from "react-router-dom";
import { getAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import { useSaasSession } from "../SaasPanels";
import { useObjectivesAndTimeline } from "../hooks/useObjectivesAndTimeline";
import type { Objective } from "../objectivesApi";
import type { TimelineEvent } from "../timelineApi";

const STATUS_LABELS: Record<Objective["status"], string> = {
  on_track: "Sur la bonne voie",
  at_risk: "À risque",
  blocked: "Bloqué",
  done: "Terminé",
};

const STATUS_BADGE_CLASS: Record<Objective["status"], string> = {
  on_track: "bg-accent-soft text-accent-primary",
  at_risk: "bg-warning/10 text-warning",
  blocked: "bg-danger/10 text-danger",
  done: "bg-accent-soft text-accent-primary",
};

const TIMELINE_ICON: Record<TimelineEvent["type"], string> = {
  email: "✉️",
  summary_update: "📝",
  action_suggested: "✅",
  meeting: "📅",
};

// Page "Objectifs" — première surface frontend pour Objectives (Phase 11) et
// Timeline (Phase 12), toutes deux fonctionnelles côté backend depuis
// Units 54-55 mais jamais appelées depuis frontend/src avant cette unité —
// même situation que Memory/Knowledge avant MemoryPage.tsx (Unit 53).
export default function ObjectivesPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);
  const {
    objectives,
    timeline,
    loading,
    error,
    addObjective,
    toggleObjectiveBlocked,
    addObjectiveMilestone,
    toggleMilestone,
  } = useObjectivesAndTimeline(saasEnabled);
  const [newTitle, setNewTitle] = useState("");
  const [milestoneDrafts, setMilestoneDrafts] = useState<Record<string, string>>({});

  if (!saasEnabled) {
    return <Navigate to="/" replace />;
  }
  if (!getAccessToken()) {
    return <Navigate to="/login" replace />;
  }
  if (!me) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-bg-primary px-4">
        <p className="text-sm text-text-muted">Chargement du compte…</p>
      </div>
    );
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    const title = newTitle.trim();
    if (!title) return;
    setNewTitle("");
    await addObjective(title);
  }

  async function handleAddMilestone(objectiveId: string) {
    const title = (milestoneDrafts[objectiveId] ?? "").trim();
    if (!title) return;
    setMilestoneDrafts((prev) => ({ ...prev, [objectiveId]: "" }));
    await addObjectiveMilestone(objectiveId, title);
  }

  return (
    <AppShell me={me}>
      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Objectifs</h1>
        <p className="mt-1 text-sm text-text-muted">
          Vos buts en cours et ce qui s'est passé récemment sur vos projets.
        </p>
      </header>

      {error ? <p className="mb-4 text-sm text-danger">{error}</p> : null}
      {loading ? <p className="text-sm text-text-muted">Chargement…</p> : null}

      {!loading ? (
        <div className="grid gap-6 lg:grid-cols-2">
          <section className="rounded-lg border border-border-default bg-surface p-4">
            <h2 className="mb-3 text-sm font-semibold text-text-primary">
              Objectifs ({objectives.length})
            </h2>

            <form onSubmit={handleCreate} className="mb-4 flex gap-2">
              <input
                type="text"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder="Nouvel objectif…"
                className="flex-1 rounded-lg border border-border-default bg-bg-tertiary/40 px-3 py-1.5 text-sm text-text-primary placeholder:text-text-muted"
              />
              <button
                type="submit"
                className="shrink-0 rounded-lg bg-accent-primary px-3 py-1.5 text-xs font-semibold text-white"
              >
                Ajouter
              </button>
            </form>

            {objectives.length === 0 ? (
              <p className="text-sm text-text-muted">Aucun objectif pour l'instant.</p>
            ) : (
              <ul className="flex flex-col gap-3">
                {objectives.map((objective) => (
                  <li
                    key={objective.id}
                    className="rounded-lg border border-border-default bg-bg-tertiary/40 p-3"
                  >
                    <div className="mb-1.5 flex items-start justify-between gap-2">
                      <p className="text-sm font-medium text-text-primary">{objective.title}</p>
                      <span
                        className={`shrink-0 rounded-full px-1.5 py-0.5 text-[11px] font-semibold ${STATUS_BADGE_CLASS[objective.status]}`}
                      >
                        {STATUS_LABELS[objective.status]}
                      </span>
                    </div>

                    {objective.blocking_reason ? (
                      <p className="mb-1.5 text-xs text-danger">{objective.blocking_reason}</p>
                    ) : null}

                    <div className="mb-2 h-1.5 w-full overflow-hidden rounded-full bg-bg-tertiary">
                      <div
                        className="h-full rounded-full bg-accent-primary"
                        style={{ width: `${objective.percent}%` }}
                      />
                    </div>
                    <p className="mb-2 text-xs text-text-muted">
                      {objective.completed_count}/{objective.total_count} jalons — {objective.percent}%
                    </p>

                    {objective.milestones.length > 0 ? (
                      <ul className="mb-2 flex flex-col gap-1">
                        {objective.milestones.map((milestone) => (
                          <li key={milestone.id} className="flex items-center gap-2 text-sm">
                            <input
                              type="checkbox"
                              checked={milestone.status === "done"}
                              onChange={() =>
                                void toggleMilestone(objective.id, milestone.id, milestone.status)
                              }
                            />
                            <span
                              className={
                                milestone.status === "done"
                                  ? "text-text-muted line-through"
                                  : "text-text-primary"
                              }
                            >
                              {milestone.title}
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : null}

                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={milestoneDrafts[objective.id] ?? ""}
                        onChange={(e) =>
                          setMilestoneDrafts((prev) => ({ ...prev, [objective.id]: e.target.value }))
                        }
                        placeholder="Nouveau jalon…"
                        className="flex-1 rounded-lg border border-border-default bg-surface px-2 py-1 text-xs text-text-primary placeholder:text-text-muted"
                      />
                      <button
                        type="button"
                        onClick={() => void handleAddMilestone(objective.id)}
                        className="shrink-0 rounded-lg px-2 py-1 text-xs text-text-secondary hover:bg-bg-tertiary"
                      >
                        Ajouter
                      </button>
                      <button
                        type="button"
                        onClick={() => void toggleObjectiveBlocked(objective)}
                        className="shrink-0 rounded-lg px-2 py-1 text-xs text-text-secondary hover:bg-bg-tertiary hover:text-danger"
                      >
                        {objective.status === "blocked" ? "Débloquer" : "Marquer bloqué"}
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="rounded-lg border border-border-default bg-surface p-4">
            <h2 className="mb-3 text-sm font-semibold text-text-primary">
              Timeline ({timeline.length})
            </h2>
            {timeline.length === 0 ? (
              <p className="text-sm text-text-muted">Aucun événement récent.</p>
            ) : (
              <ul className="flex flex-col gap-2">
                {timeline.map((event, idx) => (
                  <li
                    key={`${event.type}-${event.project_id}-${event.at}-${idx}`}
                    className="flex items-start gap-2 rounded-lg border border-border-default bg-bg-tertiary/40 p-3"
                  >
                    <span className="shrink-0">{TIMELINE_ICON[event.type]}</span>
                    <div>
                      <p className="text-sm text-text-primary">{event.label}</p>
                      <p className="text-xs text-text-muted">
                        {event.project_name} · {new Date(event.at).toLocaleString("fr-FR")}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      ) : null}
    </AppShell>
  );
}
