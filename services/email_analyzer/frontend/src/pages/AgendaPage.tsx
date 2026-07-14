import { useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import {
  fetchAgenda,
  pollAgendaRefreshJob,
  requestAgendaRefresh,
  updateAppointmentStatus,
  type AgendaOut,
  type AgendaProjectOut,
  type AppointmentOut,
} from "../agendaApi";
import { fetchActions, type ActionOut } from "../actionsApi";
import { getAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import { useSaasSession } from "../SaasPanels";

function formatDeadline(iso: string): string {
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      weekday: "long",
      day: "2-digit",
      month: "long",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

function formatDateTime(iso: string): string {
  try {
    return new Intl.DateTimeFormat("fr-FR", {
      weekday: "short",
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(iso));
  } catch {
    return iso;
  }
}

// "il y a Xh" — reflète la cadence du cron run_agenda_refresh
// (analysis_tasks.py, config.agenda_refresh_cron_hours) côté UI.
function formatRelative(iso: string): string {
  const diffMs = Date.now() - new Date(iso).getTime();
  const minutes = Math.round(diffMs / 60_000);
  if (minutes < 1) return "à l'instant";
  if (minutes < 60) return `il y a ${minutes} min`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `il y a ${hours} h`;
  const days = Math.round(hours / 24);
  return `il y a ${days} j`;
}

function AppointmentRow({
  appointment,
  onStatusChange,
}: {
  appointment: AppointmentOut;
  onStatusChange: (id: string, status: AppointmentOut["status"]) => void;
}) {
  const [busy, setBusy] = useState(false);

  async function handle(status: AppointmentOut["status"]) {
    setBusy(true);
    try {
      await updateAppointmentStatus(appointment.id, status);
      onStatusChange(appointment.id, status);
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="rounded-xl border border-border-default bg-surface px-4 py-3 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide text-accent-primary">
            {formatDateTime(appointment.scheduled_at)}
          </p>
          <p className="mt-1 text-sm text-text-primary">{appointment.description}</p>
          <p className="mt-0.5 text-xs text-text-muted">
            {appointment.project_name}
            {appointment.participants.length > 0 ? ` · ${appointment.participants.join(", ")}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 gap-1.5">
          {appointment.status !== "confirmed" ? (
            <button
              type="button"
              disabled={busy}
              onClick={() => void handle("confirmed")}
              className="rounded-md bg-success/10 px-2 py-1 text-xs font-medium text-success hover:bg-success/20 disabled:opacity-50"
            >
              Confirmer
            </button>
          ) : null}
          <button
            type="button"
            disabled={busy}
            onClick={() => void handle("cancelled")}
            className="rounded-md bg-bg-tertiary px-2 py-1 text-xs font-medium text-text-muted hover:bg-danger/10 hover:text-danger disabled:opacity-50"
          >
            Annuler
          </button>
        </div>
      </div>
    </li>
  );
}

function ProbableContactRow({ project, tone }: { project: AgendaProjectOut; tone: "warning" | "danger" }) {
  const toneClass = tone === "danger" ? "bg-danger/10 text-danger" : "bg-warning/10 text-warning";
  return (
    <li className="rounded-xl border border-border-default bg-surface px-4 py-3 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <Link
            to={`/projects/${project.project_id}`}
            className="text-sm font-semibold text-text-primary hover:text-accent-primary hover:underline"
          >
            {project.project_name}
          </Link>
          <p className="mt-1 text-xs text-text-secondary">
            {project.probable_next_contact_reason ?? "Aucune estimation disponible pour l'instant."}
          </p>
        </div>
        <span className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold ${toneClass}`}>
          {project.probable_next_contact_date
            ? formatDeadline(project.probable_next_contact_date)
            : "Aucune estimation"}
        </span>
      </div>
    </li>
  );
}

export default function AgendaPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);
  const [agenda, setAgenda] = useState<AgendaOut | null>(null);
  const [actions, setActions] = useState<ActionOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);

  async function loadAgenda() {
    const [agendaData, actionRows] = await Promise.all([fetchAgenda(), fetchActions("pending")]);
    setAgenda(agendaData);
    setActions(actionRows.filter((a) => a.deadline !== null));
  }

  useEffect(() => {
    if (!me) return;
    let cancelled = false;
    void (async () => {
      setLoading(true);
      setError(null);
      try {
        await loadAgenda();
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : "Erreur");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [me]);

  function handleAppointmentStatusChange(id: string, status: AppointmentOut["status"]) {
    setAgenda((prev) =>
      prev
        ? {
            ...prev,
            appointments:
              status === "cancelled"
                ? prev.appointments.filter((a) => a.id !== id)
                : prev.appointments.map((a) => (a.id === id ? { ...a, status } : a)),
          }
        : prev,
    );
  }

  async function handleRefresh() {
    setRefreshError(null);
    setRefreshing(true);
    try {
      const { job_id } = await requestAgendaRefresh();
      await pollAgendaRefreshJob(job_id);
      await loadAgenda();
    } catch (e) {
      setRefreshError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setRefreshing(false);
    }
  }

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

  return (
    <AppShell me={me}>
      <header
        className={`mb-6 flex items-center justify-between gap-4 rounded-2xl border p-4 transition ${
          refreshing ? "animate-fasttrack-pulse border-fasttrack-active bg-fasttrack-bg" : "border-transparent"
        }`}
      >
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Agenda</h1>
          <p className="mt-1 text-sm text-text-muted">
            Rendez-vous, prochains retours probables et échéances, tous projets confondus.
            {agenda?.agenda_updated_at ? ` · Dernière préparation IA : ${formatRelative(agenda.agenda_updated_at)}` : ""}
          </p>
        </div>
        <button
          type="button"
          disabled={refreshing}
          onClick={() => void handleRefresh()}
          className="shrink-0 rounded-lg bg-accent-primary px-3 py-2 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-50"
        >
          {refreshing ? "Rafraîchissement…" : "Rafraîchir"}
        </button>
      </header>

      {error || refreshError ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          {error ?? refreshError}
        </div>
      ) : null}

      {loading ? (
        <p className="text-sm text-text-muted">Chargement…</p>
      ) : (
        <div className="max-w-2xl space-y-8">
          <section>
            <h2 className="mb-2 text-sm font-semibold text-text-primary">Rendez-vous à venir</h2>
            {agenda && agenda.appointments.length > 0 ? (
              <ul className="space-y-2">
                {agenda.appointments.map((appt) => (
                  <AppointmentRow
                    key={appt.id}
                    appointment={appt}
                    onStatusChange={handleAppointmentStatusChange}
                  />
                ))}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">Aucun rendez-vous à venir.</p>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-text-primary">
              Projets en attente — prochain retour probable
            </h2>
            {agenda && agenda.awaiting_projects.length > 0 ? (
              <ul className="space-y-2">
                {agenda.awaiting_projects.map((p) => (
                  <ProbableContactRow key={p.project_id} project={p} tone="warning" />
                ))}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">Aucun projet en attente de retour.</p>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-text-primary">
              Projets en rouge — prochain retour attendu
            </h2>
            {agenda && agenda.at_risk_projects.length > 0 ? (
              <ul className="space-y-2">
                {agenda.at_risk_projects.map((p) => (
                  <ProbableContactRow key={p.project_id} project={p} tone="danger" />
                ))}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">Aucun projet en rouge pour l'instant.</p>
            )}
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-text-primary">Échéances (actions)</h2>
            {actions.length > 0 ? (
              <ul className="space-y-2">
                {actions.map((a) => (
                  <li
                    key={a.id}
                    className="rounded-xl border border-border-default bg-surface px-4 py-3 shadow-sm"
                  >
                    <p className="text-xs font-medium uppercase tracking-wide text-accent-primary">
                      {a.deadline ? formatDeadline(a.deadline) : ""}
                    </p>
                    <p className="mt-1 text-sm text-text-primary">{a.description}</p>
                    <p className="mt-0.5 text-xs text-text-muted">{a.project_name}</p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-text-muted">Aucune échéance à venir.</p>
            )}
          </section>
        </div>
      )}
    </AppShell>
  );
}
