import { useState } from "react";
import { Link } from "react-router-dom";
import { RunModal } from "../components/RunModal";
import { RunResult, isBlockedRun } from "../api";
import { TASK_STATUS_LABEL, blockedRunNotice, useTaskUi } from "../components/TaskDrawer";
import { useMonitor, useStatus } from "../hooks";
import { Badge, Button, Card, Empty, ErrorBox, Notice, PageHeader, formatDate, statusTone } from "../ui";

const KIND_COLORS: Record<string, string> = {
  apply: "rgb(var(--color-success-rgb))",
  message_email: "rgb(var(--color-info-rgb))",
  message_linkedin: "rgb(var(--color-accent-primary-rgb))",
  run: "rgb(var(--color-text-muted-rgb))",
};
const KIND_LABELS: Record<string, string> = {
  apply: "Candidatures", message_email: "Emails", message_linkedin: "Messages LinkedIn", run: "Analyses",
  profile_skill: "Compétence ajoutée sur LinkedIn",
};

function QuotaBar({ label, used, limit }: { label: string; used: number; limit: number }) {
  const ratio = limit > 0 ? Math.min(used / limit, 1) : 1;
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs"><span>{label}</span><span className="text-text-secondary">{used} / {limit}</span></div>
      <div className="h-2 rounded-full bg-bg-tertiary" role="progressbar" aria-valuenow={used} aria-valuemin={0} aria-valuemax={limit} aria-label={label}>
        <div className={`h-2 rounded-full ${ratio >= 1 ? "bg-danger" : ratio > 0.7 ? "bg-warning" : "bg-accent-primary"}`} style={{ width: `${ratio * 100}%` }} />
      </div>
    </div>
  );
}

function ActivityChart({ data }: { data: { date: string; [k: string]: number | string }[] }) {
  const kinds = Object.keys(KIND_COLORS);
  const totals = data.map((d) => kinds.reduce((sum, k) => sum + (Number(d[k]) || 0), 0));
  const max = Math.max(1, ...totals);
  const width = 560, height = 120, gap = 4, bar = (width - gap * (data.length - 1)) / data.length;
  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height + 18}`} className="w-full" role="img" aria-label="Activité des 14 derniers jours">
        {data.map((d, i) => {
          let y = height;
          return (
            <g key={d.date}>
              {kinds.map((k) => {
                const n = Number(d[k]) || 0;
                if (!n) return null;
                const h = (n / max) * (height - 4);
                y -= h;
                return <rect key={k} x={i * (bar + gap)} y={y} width={bar} height={h} rx={2} fill={KIND_COLORS[k]}><title>{`${d.date} · ${KIND_LABELS[k]} : ${n}`}</title></rect>;
              })}
              {i % 2 === 0 && <text x={i * (bar + gap) + bar / 2} y={height + 13} textAnchor="middle" fontSize="9" fill="rgb(var(--color-text-muted-rgb))">{String(d.date).slice(5)}</text>}
            </g>
          );
        })}
      </svg>
      <div className="mt-1 flex flex-wrap gap-3 text-xs text-text-secondary">
        {kinds.map((k) => <span key={k} className="flex items-center gap-1"><i className="inline-block h-2 w-2 rounded-sm" style={{ background: KIND_COLORS[k] }} />{KIND_LABELS[k]}</span>)}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const { data: monitor, error } = useMonitor();
  const { data: status } = useStatus();
  const { track } = useTaskUi();
  const [running, setRunning] = useState(false);

  const funnelMax = Math.max(1, ...(monitor?.funnel.map((f) => f.count) ?? [1]));
  // La dernière analyse vient-elle d'une collecte interrompue (E-4 option A) ?
  const lastReport = monitor?.last_analysis?.report;
  const lastBlocked = lastReport
    ? monitor?.recent_tasks.find((t) => isBlockedRun(t) && (t.result as RunResult | null)?.report === lastReport)
    : undefined;

  return (
    <>
      <PageHeader
        title="Tableau de bord"
        subtitle="Suivi de l'activité, des quotas et des tâches en cours"
        actions={<Button variant="primary" onClick={() => setRunning(true)}>Lancer une analyse</Button>}
      />
      <ErrorBox error={error} />
      {running && <RunModal onClose={() => setRunning(false)} />}
      {!monitor ? <p className="text-sm text-text-muted">Chargement…</p> : (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card title="Connexion LinkedIn">
            {status?.session.age_hours == null ? (
              <p className="text-sm text-text-secondary">Aucune session enregistrée : la prochaine tâche se connectera avec les identifiants du .env.</p>
            ) : (
              <p className="text-sm">
                <Badge tone={status.session.valid ? "success" : "warning"}>{status.session.valid ? "Session active" : "Session expirée"}</Badge>
                <span className="ml-2 text-text-secondary">{status.session.age_hours} h sur {status.session.ttl_hours} h</span>
              </p>
            )}
            <dl className="mt-3 space-y-1 text-xs text-text-secondary">
              <div className="flex justify-between"><dt>Identifiants (.env)</dt><dd>{status?.credentials_configured ? "configurés" : "absents"}</dd></div>
              <div className="flex justify-between"><dt>SMTP</dt><dd>{status?.smtp_configured ? "configuré" : "non configuré"}</dd></div>
              <div className="flex justify-between"><dt>Navigateur</dt><dd>{status?.browser}{status?.headless ? " (masqué)" : ""}</dd></div>
            </dl>
          </Card>

          <Card title="Quotas du jour">
            <div className="space-y-3">
              <QuotaBar label="Candidatures" used={monitor.quotas.apply.used} limit={monitor.quotas.apply.limit} />
              <QuotaBar label="Messages (email + LinkedIn)" used={monitor.quotas.messages.used} limit={monitor.quotas.messages.limit} />
            </div>
            <p className="mt-3 text-xs text-text-muted">Les plafonds se règlent dans Paramètres.</p>
          </Card>

          <Card title="Dernière analyse" actions={<Link className="text-xs text-accent-primary" to="/marche">Voir le marché</Link>}>
            {monitor.last_analysis ? (
              <div className="space-y-1 text-sm">
                {lastBlocked && <div className="mb-2"><Notice tone="warning">{blockedRunNotice(lastBlocked)}</Notice></div>}
                <p className="font-medium">{monitor.last_analysis.keyword}</p>
                <p className="text-text-secondary">{formatDate(monitor.last_analysis.generated_at)}</p>
                <p className="text-text-secondary">{monitor.last_analysis.counts.job} offres · {monitor.last_analysis.counts.post} posts · {monitor.last_analysis.counts.person} profils</p>
                {monitor.model && (
                  <p className="text-text-secondary">
                    Modèle : {monitor.model.trained ? `entraîné sur ${monitor.model.n_labels} annotations (${Math.round((monitor.model.train_accuracy ?? 0) * 100)} %)` : `poids a priori (${monitor.model.n_labels} annotations)`}
                  </p>
                )}
              </div>
            ) : <Empty>Aucune analyse pour l'instant.</Empty>}
          </Card>

          <Card title="Entonnoir de candidature" className="lg:col-span-2">
            <ul className="space-y-2">
              {monitor.funnel.map((step) => (
                <li key={step.step}>
                  <div className="mb-0.5 flex justify-between text-xs"><span>{step.step}</span><span className="font-medium">{step.count}</span></div>
                  <div className="h-2.5 rounded-full bg-bg-tertiary"><div className="h-2.5 rounded-full bg-accent-primary" style={{ width: `${(step.count / funnelMax) * 100}%` }} /></div>
                </li>
              ))}
            </ul>
            <div className="mt-3 flex flex-wrap gap-2">
              {Object.entries(monitor.by_status).map(([s, n]) => <Badge key={s} tone={statusTone(s)}>{s} : {n}</Badge>)}
            </div>
          </Card>

          <Card title="Tâches en cours">
            {monitor.active_tasks.length === 0 ? <p className="text-sm text-text-muted">Aucune tâche active.</p> : (
              <ul className="space-y-2">
                {monitor.active_tasks.map((t) => (
                  <li key={t.id}>
                    <button className="w-full rounded-md border border-border-default p-2 text-left hover:bg-bg-secondary" onClick={() => track(t)}>
                      <div className="flex justify-between gap-2 text-sm"><span className="truncate">{t.title}</span><Badge tone="info">{TASK_STATUS_LABEL[t.status]}</Badge></div>
                      <div className="mt-1 truncate font-mono text-xs text-text-muted">{t.logs[t.logs.length - 1]}</div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Activité (14 jours)" className="lg:col-span-2"><ActivityChart data={monitor.activity} /></Card>

          <Card title="Erreurs récentes">
            {monitor.errors.length === 0 ? <p className="text-sm text-text-muted">Aucune erreur.</p> : (
              <ul className="space-y-2">
                {monitor.errors.map((e) => (
                  <li key={e.id} className="text-sm"><span className="font-medium">{e.title}</span><br /><span className="text-xs text-danger">{e.error}</span></li>
                ))}
              </ul>
            )}
          </Card>

          <Card title="Derniers événements" className="lg:col-span-3">
            {monitor.recent_events.length === 0 ? <p className="text-sm text-text-muted">Rien pour l'instant.</p> : (
              <ul className="divide-y divide-border-subtle text-sm">
                {monitor.recent_events.map((e, i) => (
                  <li key={i} className="flex gap-3 py-1.5">
                    <span className="w-32 shrink-0 text-xs text-text-muted">{formatDate(e.at)}</span>
                    <Badge>{KIND_LABELS[e.kind] ?? e.kind}</Badge>
                    <span className="truncate text-text-secondary">{String(e.title ?? e.keyword ?? e.to ?? "")}</span>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}
      {status && !status.has_analysis && <div className="mt-4"><Notice>Lance une première analyse pour alimenter les offres, le marché et les suggestions de compétences.</Notice></div>}
    </>
  );
}
