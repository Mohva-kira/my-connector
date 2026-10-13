import { CSSProperties, ReactNode, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Job, Report, SkillAdded, SourceCode, Task, api, authedUrl, enc, sourceLabel } from "../api";
import { Icon, IconName } from "../components/icons";
import { RunModal } from "../components/RunModal";
import { useTaskUi } from "../components/TaskDrawer";
import { useAction, useLabelJob, useProfile, useReport, useReports } from "../hooks";
import { Badge, Button, Empty, ErrorBox, Notice, formatDate, inputClass, scoreTone } from "../ui";
import { JobDetailModal } from "./Jobs";

const reducedMotion = () => typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
const stagger = (i: number) => ({ "--i": i }) as CSSProperties;

/** Valeur qui monte de 0 à `target` à l'affichage (immédiate si les animations sont réduites). */
function useCountUp(target: number, duration = 700) {
  const reduced = reducedMotion();
  const [value, setValue] = useState(reduced ? target : 0);
  useEffect(() => {
    if (reduced) return;
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      setValue(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [target, duration, reduced]);
  return reduced ? target : value;
}

// ------------------------------------------------------------------ primitives de la page
function Panel({ title, icon, actions, children, className = "", index = 0 }: {
  title?: ReactNode; icon?: IconName; actions?: ReactNode; children: ReactNode; className?: string; index?: number;
}) {
  return (
    <section style={stagger(index)} className={`animate-in min-w-0 rounded-xl border border-border-default bg-surface p-4 shadow-card sm:p-5 ${className}`}>
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-center justify-between gap-2">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-text-primary">
            {icon && <span className="grid h-7 w-7 place-items-center rounded-md bg-accent-soft text-accent-primary"><Icon name={icon} /></span>}
            {title}
          </h3>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

function SectionHeading({ id, title, hint, actions }: { id: string; title: string; hint?: string; actions?: ReactNode }) {
  return (
    <div id={id} className="mb-3 mt-10 flex scroll-mt-20 flex-wrap items-end justify-between gap-2 first:mt-0">
      <div>
        <h2 className="text-base font-semibold">{title}</h2>
        {hint && <p className="text-sm text-text-secondary">{hint}</p>}
      </div>
      {actions}
    </div>
  );
}

const RING_TONE = { success: "--color-success-rgb", warning: "--color-warning-rgb", neutral: "--color-text-muted-rgb" } as const;

function ScoreRing({ score, size = 52 }: { score: number; size?: number }) {
  const r = size / 2 - 4;
  const length = 2 * Math.PI * r;
  const tone = scoreTone(score) as keyof typeof RING_TONE;
  const color = `rgb(var(${RING_TONE[tone] ?? RING_TONE.neutral}))`;
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }} aria-label={`Score ${Math.round(score)} %`}>
      <svg viewBox={`0 0 ${size} ${size}`} className="-rotate-90" width={size} height={size} aria-hidden="true">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(var(--color-bg-tertiary-rgb))" strokeWidth="4" />
        <circle
          className="ring-draw" cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="4" strokeLinecap="round"
          strokeDasharray={length} strokeDashoffset={length * (1 - Math.min(100, score) / 100)} style={{ "--ring-length": length } as CSSProperties}
        />
      </svg>
      <span className="absolute inset-0 grid place-items-center text-xs font-semibold">{Math.round(score)}</span>
    </div>
  );
}

/** Message d'un graphique ou d'une section sans données (rapport ancien ou incomplet). */
function Insufficient() {
  return <p className="text-sm text-text-muted">Données insuffisantes.</p>;
}

function Bars({ rows, tone = "bg-accent-primary" }: {
  rows: { label: string; value: number; extra?: ReactNode; muted?: boolean; action?: ReactNode }[]; tone?: string;
}) {
  if (rows.length === 0) return <Insufficient />;
  const max = rows.reduce((m, r) => Math.max(m, r.value), 0.0001);
  return (
    <ul className="space-y-2">
      {rows.map((r, i) => (
        <li key={r.label} className="group flex items-center gap-2 text-sm sm:gap-3">
          <span className={`w-24 shrink-0 truncate sm:w-36 ${r.muted ? "text-text-secondary" : "text-text-primary"}`} title={r.label}>{r.label}</span>
          <span className="h-2 min-w-8 flex-1 overflow-hidden rounded-full bg-bg-tertiary">
            <span style={{ ...stagger(i), width: `${(r.value / max) * 100}%` }} className={`bar-grow block h-2 rounded-full ${r.muted ? "bg-text-muted/40" : tone}`} />
          </span>
          <span className="w-16 shrink-0 text-right text-xs text-text-secondary sm:w-20">{r.extra}</span>
          {r.action !== undefined && <span className="w-16 shrink-0 text-right sm:w-20">{r.action}</span>}
        </li>
      ))}
    </ul>
  );
}

function WeeklyChart({ weeks: allWeeks, counts: allCounts }: { weeks: string[]; counts: number[] }) {
  // Semaines et effectifs appariés : un rapport incomplet ne doit ni planter ni décaler les libellés.
  const n = Math.min(allWeeks.length, allCounts.length);
  const weeks = allWeeks.slice(0, n);
  const counts = allCounts.slice(0, n).map((c) => (Number.isFinite(c) ? c : 0));
  if (n === 0) return <Insufficient />;
  const max = counts.reduce((m, c) => Math.max(m, c), 1);
  const w = 360, h = 90, step = counts.length > 1 ? w / (counts.length - 1) : w;
  const y = (c: number) => h - (c / max) * (h - 8) - 4;
  const points = counts.map((c, i) => `${i * step},${y(c)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h + 16}`} className="w-full overflow-visible" role="img" aria-label="Éléments collectés par semaine">
      <polygon points={`0,${h} ${points} ${(counts.length - 1) * step},${h}`} fill="rgba(var(--color-accent-primary-rgb), 0.08)" />
      <polyline points={points} fill="none" stroke="rgb(var(--color-accent-primary-rgb))" strokeWidth="2" strokeLinejoin="round" />
      {counts.map((c, i) => (
        <g key={weeks[i]}>
          <circle cx={i * step} cy={y(c)} r="3.5" fill="rgb(var(--color-surface-rgb))" stroke="rgb(var(--color-accent-primary-rgb))" strokeWidth="2"><title>{`${weeks[i]} : ${c}`}</title></circle>
          <text x={i * step} y={h + 13} textAnchor={i === 0 ? "start" : i === counts.length - 1 ? "end" : "middle"} fontSize="9" fill="rgb(var(--color-text-muted-rgb))">{weeks[i].slice(5)}</text>
        </g>
      ))}
    </svg>
  );
}

function Kpi({ label, value, suffix = "", hint, delta, index }: { label: string; value: number; suffix?: string; hint?: ReactNode; delta?: number | null; index: number }) {
  const shown = useCountUp(value);
  return (
    <div style={stagger(index)} className="animate-in card-lift rounded-xl border border-border-default bg-surface p-4 shadow-card">
      <div className="text-xs font-medium uppercase tracking-wide text-text-muted">{label}</div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className="text-2xl font-semibold tabular-nums">{Number.isInteger(value) ? Math.round(shown) : shown.toFixed(1)}{suffix}</span>
        {delta != null && delta !== 0 && (
          <span className={`text-xs font-medium ${delta > 0 ? "text-success" : "text-warning"}`}>{delta > 0 ? "+" : ""}{delta.toFixed(1)} pt</span>
        )}
      </div>
      {hint && <div className="mt-1 text-xs text-text-secondary">{hint}</div>}
    </div>
  );
}

// ------------------------------------------------------------------ interprétation découpée par titres
interface Section { title: string; body: string }

function splitSections(markdown: string): Section[] {
  const parts = markdown.split(/^#{1,3}\s+/m);
  const sections = parts.slice(1).map((part) => {
    const [heading, ...rest] = part.split("\n");
    return { title: heading.replace(/^\d+[.)]\s*/, "").trim(), body: rest.join("\n").trim() };
  });
  const intro = parts[0].trim();
  return intro ? [{ title: "Synthèse", body: intro }, ...sections] : sections;
}

const isOffersSection = (title: string) => /offres?/i.test(title);

function sectionIcon(title: string): IconName {
  const t = title.toLowerCase();
  if (t.includes("lecture") || t.includes("synth")) return "chart";
  if (t.includes("tendance")) return "trend";
  if (t.includes("écart") || t.includes("ecart")) return "target";
  if (t.includes("matching") || t.includes("pèse")) return "scale";
  if (t.includes("ajust") || t.includes("profil")) return "user";
  if (t.includes("limite")) return "info";
  return "spark";
}

// ------------------------------------------------------------------ offres à cibler
function shortApplicants(text?: string) {
  return text?.replace("personnes ont cliqué sur Postuler", "clics « Postuler »");
}

function OfferCard({ job, index, actionable, onOpen }: { job: Job; index: number; actionable: boolean; onOpen: () => void }) {
  const { track } = useTaskUi();
  const navigate = useNavigate();
  const label = useLabelJob();
  const prepare = useAction(() => api<Task>(`/jobs/${enc(job.id)}/prepare`, { method: "POST" }), [["tasks"]]);
  const crowded = job.applicants?.startsWith("Plus de");

  return (
    <article style={stagger(index)} className="animate-in card-lift flex flex-col rounded-xl border border-border-default bg-surface p-4 shadow-card hover:border-accent-primary/40">
      <div className="flex items-start gap-3">
        <ScoreRing score={job.score} />
        <div className="min-w-0 flex-1">
          <button onClick={onOpen} disabled={!actionable} className="text-left font-semibold leading-snug hover:text-accent-primary disabled:hover:text-text-primary">
            {job.title || "Sans titre"}
          </button>
          <div className="mt-0.5 truncate text-sm text-text-secondary">{job.company || "Entreprise inconnue"}</div>
        </div>
        {job.application_status && <Badge tone="accent">{job.application_status}</Badge>}
      </div>

      <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-xs text-text-secondary">
        {job.location && <span className="flex items-center gap-1"><Icon name="pin" className="h-3.5 w-3.5" />{job.location}</span>}
        {(job.posted_label || job.posted_at) && <span className="flex items-center gap-1"><Icon name="clock" className="h-3.5 w-3.5" />{job.posted_label || job.posted_at}</span>}
        {job.applicants && (
          <span className={`flex items-center gap-1 ${crowded ? "text-warning" : ""}`} title="Concurrence sur l'offre"><Icon name="users" className="h-3.5 w-3.5" />{shortApplicants(job.applicants)}</span>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        <Badge>{sourceLabel(job.source_code)}</Badge>
        {job.remote && <Badge tone="info">Remote</Badge>}
        {job.freelance && <Badge tone="info">Freelance</Badge>}
        {job.matched_skills.slice(0, 4).map((s) => <Badge key={s} tone="success">✓ {s}</Badge>)}
        {job.missing_skills.slice(0, 3).map((s) => <Badge key={s}>{s}</Badge>)}
        {job.missing_skills.length > 3 && <span className="self-center text-xs text-text-muted">+{job.missing_skills.length - 3}</span>}
      </div>

      <ErrorBox error={label.error ?? prepare.error} />

      <div className="min-h-4 flex-1" />
      <footer className="flex flex-wrap items-center gap-1.5 border-t border-border-subtle pt-3">
        <div className="flex gap-1" role="group" aria-label="Annotation">
          <button
            type="button" title="Pertinente" aria-label="Pertinente" aria-pressed={job.label === 1} onClick={() => label.mutate({ id: job.id, value: job.label === 1 ? null : 1 })}
            className={`rounded-md p-1.5 transition ${job.label === 1 ? "bg-success/15 text-success" : "text-text-muted hover:bg-bg-secondary hover:text-text-primary"}`}
          ><Icon name="thumbUp" /></button>
          <button
            type="button" title="Pas pertinente" aria-label="Pas pertinente" aria-pressed={job.label === 0} onClick={() => label.mutate({ id: job.id, value: job.label === 0 ? null : 0 })}
            className={`rounded-md p-1.5 transition ${job.label === 0 ? "bg-warning/15 text-warning" : "text-text-muted hover:bg-bg-secondary hover:text-text-primary"}`}
          ><Icon name="thumbDown" /></button>
        </div>
        <a href={job.url} target="_blank" rel="noopener noreferrer" title="Ouvrir l'offre"
          aria-label={job.title ? `Ouvrir l'offre « ${job.title} » sur ${sourceLabel(job.source_code)} (nouvel onglet)` : `Ouvrir l'offre sur ${sourceLabel(job.source_code)} (nouvel onglet)`}
          className="rounded-md p-1.5 text-text-muted transition hover:bg-bg-secondary hover:text-text-primary">
          <Icon name="external" />
        </a>
        <div className="ml-auto flex gap-1.5">
          {actionable && <Button small variant="ghost" onClick={onOpen}>Détail</Button>}
          {actionable && (job.application_status ? (
            <Button small onClick={() => navigate(`/candidatures/${enc(job.id)}`)}>Voir la candidature</Button>
          ) : (
            <Button small variant="primary" loading={prepare.isPending} onClick={() => prepare.mutate(undefined, { onSuccess: track })}>Préparer</Button>
          ))}
        </div>
      </footer>
    </article>
  );
}

// ------------------------------------------------------------------ page
const NAV = [
  { id: "synthese", label: "Synthèse" },
  { id: "offres", label: "Offres à cibler" },
  { id: "competences", label: "Compétences" },
  { id: "analyse", label: "Analyse" },
  { id: "signaux", label: "Signaux" },
];

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      setTimeout(() => setDone(false), 1800);
    } catch { /* presse-papiers refusé : rien à faire */ }
  };
  return <Button onClick={copy} title="Copier l'analyse (Markdown)"><span className="flex items-center gap-1.5"><Icon name={done ? "check" : "copy"} />{done ? "Copiée" : "Copier"}</span></Button>;
}

export default function Market() {
  const { data: reports, error: reportsError } = useReports();
  // Rapport choisi dans l'URL (`/marche?rapport={name}`) : lien direct et bouton Retour cohérent.
  const [params, setParams] = useSearchParams();
  const chosen = params.get("rapport") ?? "";
  const chooseReport = (value: string) => setParams((current) => {
    const next = new URLSearchParams(current);
    next.set("rapport", value);
    return next;
  });
  const [running, setRunning] = useState(false);
  const [openJob, setOpenJob] = useState<string | null>(null);
  // « Voir toutes les offres » vaut pour un rapport donné : changer de rapport (y compris par Retour) replie la liste.
  const [showAllFor, setShowAllFor] = useState<string | null>(null);
  // Filtre de source de la liste d'offres, propre à un rapport : changer de rapport le réinitialise.
  const [sourceFilter, setSourceFilter] = useState<{ report: string | null; value: SourceCode | "" }>({ report: null, value: "" });
  const name = chosen || reports?.[0]?.name;
  const showAllJobs = !!name && showAllFor === name;
  const sourceValue = sourceFilter.report === (name ?? null) ? sourceFilter.value : "";
  const row = reports?.find((r) => r.name === name);
  const { data: report, error } = useReport(name);
  // Rapport précédent du même mot-clé, pour l'évolution de la couverture.
  const previousName = useMemo(() => {
    if (!reports || !row) return undefined;
    const older = reports.slice(reports.indexOf(row) + 1);
    return older.find((r) => r.keyword.toLowerCase() === row.keyword.toLowerCase())?.name;
  }, [reports, row]);
  const { data: previous } = useReport(previousName);
  const { data: profile } = useProfile();
  const { track } = useTaskUi();

  const [added, setAdded] = useState<string[]>([]);
  const [syncNote, setSyncNote] = useState("");
  const addSkill = useAction((skill: string) => api<SkillAdded>("/profile/skills", { method: "POST", body: { skill } }), [["profile"], ["tasks"]]);
  const onSkillAdded = (skill: string, res: SkillAdded) => {
    setAdded((a) => [...a, skill]);
    setSyncNote(res.linkedin_sync.status === "unavailable" ? res.linkedin_sync.message ?? "" : "");
    if (res.linkedin_sync.task) track(res.linkedin_sync.task);
  };
  const rescore = useAction(() => api<Task>("/runs/rescore", { method: "POST" }), [["tasks"]]);
  const train = useAction(() => api<Task>("/model/train", { method: "POST" }), [["tasks"]]);

  const owned = useMemo(() => new Set((profile?.skills ?? []).map((s) => s.toLowerCase())), [profile?.skills]);

  return (
    <div className="sticky-nav-offset">
      <header className="animate-in mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 tabIndex={-1} className="text-xl font-semibold focus:outline-none">Marché</h1>
          <p className="mt-0.5 text-sm text-text-secondary">
            {report ? <>« {report.request.keyword} » · {formatDate(report.generated_at)}{report.interpreter && <> · interprétation {report.interpreter}</>}</> : "Tendances, compétences demandées et offres à cibler"}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {reports && reports.length > 0 && (
            <select className={`${inputClass} sm:!w-72`} value={name ?? ""} onChange={(e) => chooseReport(e.target.value)} aria-label="Rapport">
              {reports.map((r) => <option key={r.name} value={r.name}>{r.keyword} — {formatDate(r.generated_at)}</option>)}
            </select>
          )}
          {report?.interpretation && <CopyButton text={report.interpretation} />}
          {name && (
            <a className="flex items-center gap-1.5 rounded-md border border-border-default bg-surface px-3.5 py-2 text-sm hover:bg-bg-secondary" href={authedUrl(`/reports/${enc(name)}`)} download={`${name}.json`} title="Exporter le rapport (JSON)">
              <Icon name="download" />JSON
            </a>
          )}
          {row?.html && (
            <a className="flex items-center gap-1.5 rounded-md border border-border-default bg-surface px-3.5 py-2 text-sm hover:bg-bg-secondary" target="_blank" rel="noopener noreferrer" href={authedUrl(`/reports/${enc(row.name)}/html`)}>
              <Icon name="doc" />Rapport complet
            </a>
          )}
          <Button variant="primary" onClick={() => setRunning(true)}><span className="flex items-center gap-1.5"><Icon name="play" />Nouvelle analyse</span></Button>
        </div>
      </header>
      {running && <RunModal onClose={() => setRunning(false)} />}
      <ErrorBox error={reportsError ?? error ?? addSkill.error ?? rescore.error ?? train.error} />
      {syncNote && <div className="mb-4"><Notice tone="warning">{syncNote}</Notice></div>}

      {!reports ? <p className="text-sm text-text-muted">Chargement…</p> : !reports.length ? (
        <Empty>Aucun rapport pour l'instant. Lance une première analyse avec « Nouvelle analyse ».</Empty>
      ) : !report ? <div className="h-40 animate-pulse rounded-xl bg-bg-secondary" /> : (
        <MarketBody
          report={report}
          previous={previous}
          owned={owned}
          added={added}
          showAllJobs={showAllJobs}
          sourceFilter={sourceValue}
          onSourceFilter={(value) => setSourceFilter({ report: name ?? null, value })}
          onShowAllJobs={() => setShowAllFor(name ?? null)}
          onOpenJob={setOpenJob}
          onAddSkill={(skill) => addSkill.mutate(skill, { onSuccess: (res) => onSkillAdded(skill, res) })}
          addingSkill={addSkill.isPending ? addSkill.variables : undefined}
          onRescore={() => rescore.mutate(undefined, { onSuccess: (task) => { track(task); setAdded([]); } })}
          rescoring={rescore.isPending}
          onTrain={() => train.mutate(undefined, { onSuccess: track })}
          training={train.isPending}
        />
      )}
      {openJob && <JobDetailModal id={openJob} onClose={() => setOpenJob(null)} />}
    </div>
  );
}

function MarketBody({
  report, previous, owned, added, showAllJobs, sourceFilter, onSourceFilter, onShowAllJobs, onOpenJob, onAddSkill, addingSkill, onRescore, rescoring, onTrain, training,
}: {
  report: Report; previous?: Report; owned: Set<string>; added: string[]; showAllJobs: boolean;
  sourceFilter: SourceCode | ""; onSourceFilter: (value: SourceCode | "") => void; onShowAllJobs: () => void;
  onOpenJob: (id: string) => void; onAddSkill: (skill: string) => void; addingSkill?: string; onRescore: () => void; rescoring: boolean;
  onTrain: () => void; training: boolean;
}) {
  const sections = useMemo(() => splitSections(report.interpretation || ""), [report.interpretation]);
  const allJobs = report.top_jobs;
  // Filtre côté client sur la liste seulement : KPI, compteurs et analyses restent ceux du rapport entier.
  const reportSources = useMemo(() => [...new Set(allJobs.map((j) => j.source_code))], [allJobs]);
  const jobs = sourceFilter ? allJobs.filter((j) => j.source_code === sourceFilter) : allJobs;
  const visibleJobs = showAllJobs ? jobs : jobs.slice(0, 6);
  const strong = allJobs.filter((j) => j.score >= 75).length;
  const missingTop = report.gap.skills.filter((s) => !s.owned && !owned.has(s.skill.toLowerCase())).slice(0, 5);
  const coverageDelta = previous ? report.gap.coverage - previous.gap.coverage : null;
  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  return (
    <>
      <nav aria-label="Sections de la page" className="sticky top-0 z-10 -mx-4 mb-6 flex gap-1 overflow-x-auto border-b border-border-default bg-bg-primary/85 px-4 py-2 backdrop-blur md:-mx-6 md:px-6">
        {NAV.map((item) => (
          <button key={item.id} onClick={() => scrollTo(item.id)} className="whitespace-nowrap rounded-full px-3 py-1.5 text-sm text-text-secondary transition hover:bg-accent-soft hover:text-accent-primary">
            {item.label}
          </button>
        ))}
      </nav>

      {/* ---------------------------------------------------------- synthèse */}
      <SectionHeading id="synthese" title="Synthèse" />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi index={0} label="Offres analysées" value={report.counts.job ?? 0} hint={`${report.counts.person ?? 0} profils · ${report.counts.post ?? 0} posts`} />
        <Kpi index={1} label="Couverture compétences" value={report.gap.coverage} suffix=" %" delta={coverageDelta} hint={previous ? "vs analyse précédente" : "part pondérée des compétences demandées"} />
        <Kpi index={2} label="Offres ≥ 75 %" value={strong} hint={strong ? "bonnes correspondances" : "aucune forte correspondance"} />
        <Kpi index={3} label="Compétences en hausse" value={report.rising.skills.filter((s) => s.growth > 0).length} hint="sur les 15 derniers jours" />
      </div>
      {missingTop.length > 0 && (
        <div style={stagger(4)} className="animate-in mt-3 flex flex-wrap items-center gap-2 rounded-xl border border-border-default bg-accent-soft px-4 py-3 text-sm">
          <Icon name="target" className="h-4 w-4 text-accent-primary" />
          <span className="font-medium">Priorités à acquérir ou rendre visibles :</span>
          {missingTop.map((s) => <Badge key={s.skill}>{s.skill}</Badge>)}
          <button onClick={() => scrollTo("competences")} className="ml-auto flex items-center gap-1 text-accent-primary hover:underline">Gérer <Icon name="arrowRight" className="h-3.5 w-3.5" /></button>
        </div>
      )}

      {/* ---------------------------------------------------------- offres */}
      <SectionHeading
        id="offres"
        title="Offres à cibler"
        hint={`${jobs.length} offres classées par score de correspondance`}
        actions={<div className="flex flex-wrap items-center gap-3">
          {reportSources.length >= 2 && (
            <label className="flex items-center gap-2 text-sm text-text-secondary">
              Source (filtre la liste, pas les indicateurs)
              <select className={`${inputClass} !w-auto`} value={sourceFilter} onChange={(e) => onSourceFilter(e.target.value as SourceCode | "")}>
                <option value="">Toutes les sources</option>
                {reportSources.map((code) => <option key={code} value={code}>{sourceLabel(code)}</option>)}
              </select>
            </label>
          )}
          <Link to="/offres" className="flex items-center gap-1 text-sm text-accent-primary hover:underline">Toutes les offres <Icon name="arrowRight" className="h-3.5 w-3.5" /></Link>
        </div>}
      />
      {!report.is_latest && (
        <div className="mb-3"><Notice tone="info">Rapport antérieur : la préparation et le détail des offres sont disponibles depuis le dernier rapport.</Notice></div>
      )}
      {allJobs.length === 0 ? <Empty>Aucune offre dans ce rapport.</Empty> : jobs.length === 0 ? <Empty>Aucune offre de cette source dans ce rapport.</Empty> : (
        <>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {visibleJobs.map((job, i) => (
              <OfferCard key={job.id} job={job} index={i} actionable={report.is_latest} onOpen={() => onOpenJob(job.id)} />
            ))}
          </div>
          {!showAllJobs && jobs.length > visibleJobs.length && (
            <div className="mt-3 text-center">
              <Button variant="ghost" onClick={onShowAllJobs}>Afficher les {jobs.length - visibleJobs.length} autres offres</Button>
            </div>
          )}
        </>
      )}

      {/* ---------------------------------------------------------- compétences */}
      <SectionHeading id="competences" title="Compétences" hint="Ce que le marché demande, et ce qui monte" actions={<Link to="/competences" className="flex items-center gap-1 text-sm text-accent-primary hover:underline">Mon profil <Icon name="arrowRight" className="h-3.5 w-3.5" /></Link>} />
      {added.length > 0 && (
        <div className="animate-in mb-3">
          <Notice tone="success">
            <span className="flex flex-wrap items-center gap-2">
              Ajoutée{added.length > 1 ? "s" : ""} à ton profil : {added.join(", ")}. Les scores ne tiennent pas encore compte de ces changements.
              <Button small variant="primary" loading={rescoring} onClick={onRescore}><span className="flex items-center gap-1"><Icon name="refresh" className="h-3.5 w-3.5" />Recalculer les scores</span></Button>
            </span>
          </Notice>
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Compétences demandées" icon="target" index={0}>
          <Bars rows={report.gap.skills.slice(0, 15).map((s) => {
            const has = s.owned || owned.has(s.skill.toLowerCase());
            return {
              label: s.skill, value: s.weight, muted: !has,
              extra: has ? <span className="text-success">✓ acquise</span> : "manquante",
              action: has ? null : (
                <button
                  onClick={() => onAddSkill(s.skill)} disabled={addingSkill === s.skill}
                  title="Je maîtrise déjà cette compétence : l'ajouter à mon profil"
                  className="inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs text-accent-primary opacity-80 transition hover:bg-accent-soft hover:opacity-100 disabled:opacity-40"
                ><Icon name="plus" className="h-3.5 w-3.5" />J'ai</button>
              ),
            };
          })} />
        </Panel>
        <Panel title="En hausse" icon="trend" index={1} actions={<span className="text-xs text-text-muted">15 derniers jours vs précédents</span>}>
          {report.rising.skills.length === 0 ? <p className="text-sm text-text-muted">Pas assez d'historique pour dégager une tendance.</p> : (
            <Bars tone="bg-success" rows={report.rising.skills.slice(0, 12).map((s) => ({ label: s.term, value: Math.max(s.growth, 0), extra: `${s.older} → ${s.recent}` }))} />
          )}
        </Panel>
      </div>

      {/* ---------------------------------------------------------- analyse */}
      <SectionHeading id="analyse" title="Analyse" hint={report.interpreter ? `Lecture rédigée par l'interprète ${report.interpreter}` : undefined} />
      {sections.length === 0 ? <Empty>Pas d'interprétation pour ce rapport.</Empty> : (
        <div className="grid gap-4 lg:grid-cols-2">
          {sections.map((section, i) => isOffersSection(section.title) && allJobs.length > 0 ? (
            <Panel key={section.title} title={section.title} icon="target" index={i}>
              <p className="text-sm text-text-secondary">Les offres de cette section sont présentées, nettoyées et actionnables, plus haut.</p>
              <button onClick={() => scrollTo("offres")} className="mt-2 flex items-center gap-1 text-sm text-accent-primary hover:underline">Voir les offres à cibler <Icon name="arrowRight" className="h-3.5 w-3.5" /></button>
            </Panel>
          ) : (
            <Panel key={section.title} title={section.title} icon={sectionIcon(section.title)} index={i} className={i === 0 ? "lg:col-span-2" : ""}>
              <div className="prose-md text-sm leading-relaxed text-text-secondary [&_strong]:text-text-primary"><ReactMarkdown remarkPlugins={[remarkGfm]}>{section.body}</ReactMarkdown></div>
            </Panel>
          ))}
        </div>
      )}

      {/* ---------------------------------------------------------- signaux */}
      <SectionHeading id="signaux" title="Signaux" hint="Volume de collecte, vocabulaire et modèle de matching" />
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Éléments collectés par semaine" icon="chart" index={0}>
          <WeeklyChart weeks={report.weekly.weeks} counts={report.weekly.counts} />
        </Panel>
        <Panel title="Termes les plus fréquents" icon="spark" index={1}>
          <Bars rows={report.top_terms.slice(0, 10).map((t) => ({ label: t.term, value: t.share, extra: `${Math.round(t.share * 100)} %` }))} />
        </Panel>
        <Panel
          title="Modèle de matching" icon="scale" index={2} className="lg:col-span-2"
          actions={<Button small loading={training} onClick={onTrain}><span className="flex items-center gap-1"><Icon name="refresh" className="h-3.5 w-3.5" />Réentraîner</span></Button>}
        >
          <p className="mb-3 text-sm text-text-secondary">
            {report.model.trained
              ? `Appris sur ${report.model.n_labels} annotations — précision ${Math.round((report.model.train_accuracy ?? 0) * 100)} %.`
              : "Poids a priori : pas encore assez d'annotations."}{" "}
            Annote les offres (pouce haut / bas) pour affiner le classement.
          </p>
          <Bars rows={report.model.importance.map((f) => ({
            label: f.feature.replace(/_/g, " "), value: Math.abs(f.weight), muted: f.weight < 0,
            extra: <span className={f.weight >= 0 ? "text-success" : "text-text-muted"}>{f.weight >= 0 ? "+" : ""}{f.weight.toFixed(1)}</span>,
          }))} />
        </Panel>
      </div>
    </>
  );
}
