// Client de l'API locale. Le jeton est injecté dans index.html par le serveur (meta lm-token).

const token = document.querySelector('meta[name="lm-token"]')?.getAttribute("content") ?? "";

/** Requête d'origine d'une erreur (relance explicite après une popup de risques, sans rejeu automatique). */
export interface ApiRequest { path: string; method: string; body?: unknown }

/**
 * Erreur d'API. `detail` est le message serveur normalisé, affiché tel quel ; `status` vaut 0 si l'API est
 * injoignable ; `retryAfter` (secondes) vient de l'en-tête `Retry-After` des 429 (G-2), `null` s'il est absent.
 * Sur un 428 (I-16), `source` et `riskVersion` désignent la source dont les risques sont à accepter.
 */
export class ApiError extends Error {
  source: string | null = null;
  riskVersion: string | null = null;
  request: ApiRequest | null = null;

  constructor(public status: number, public detail: string, public body: unknown = null, public retryAfter: number | null = null) {
    super(detail);
    this.name = "ApiError";
    if (status === 428 && body && typeof body === "object") {
      const { source, risk_version } = body as { source?: unknown; risk_version?: unknown };
      this.source = typeof source === "string" ? source : null;
      this.riskVersion = typeof risk_version === "string" ? risk_version : null;
    }
  }
}

// ------------------------------------------------------------------ 428 : consentement de source requis
const consentListeners = new Set<(error: ApiError) => void>();

/** Abonnement aux 428 (popup de risques) ; renvoie la fonction de désabonnement. */
export function onConsentRequired(listener: (error: ApiError) => void): () => void {
  consentListeners.add(listener);
  return () => { consentListeners.delete(listener); };
}

const LOC_ROOTS = new Set(["body", "query", "path"]);

/** Une erreur Pydantic en liste (forme antérieure à G-1, encore acceptée) → « champ : message ». */
function validationItem(item: unknown): string {
  const { loc, msg } = (item ?? {}) as { loc?: unknown[]; msg?: unknown };
  const message = typeof msg === "string" ? msg.replace(/^Value error, /, "") : "valeur invalide";
  const field = Array.isArray(loc) ? [...loc].reverse().find((part) => !LOC_ROOTS.has(String(part))) : undefined;
  return field === undefined ? message : `${String(field)} : ${message}`;
}

/** `detail` normalisé (G-1) : chaîne telle quelle ; liste → « champ : message » joints par « ; ». */
export function detailOf(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map(validationItem).filter(Boolean).join(" ; ") || fallback;
  return fallback;
}

function retryAfterOf(response: Response): number | null {
  const seconds = Number(response.headers.get("Retry-After"));
  return Number.isInteger(seconds) && seconds >= 1 ? seconds : null;
}

export async function api<T>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      method: init.method ?? "GET",
      headers: { "Content-Type": "application/json", ...(token ? { "X-Local-Token": token } : {}) },
      body: init.body === undefined ? undefined : JSON.stringify(init.body),
    });
  } catch {
    throw new ApiError(0, "API locale injoignable : l'application est-elle lancée ?");
  }
  const text = await response.text();
  let parsed: unknown = null;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    /* corps non JSON */
  }
  if (!response.ok) {
    const error = new ApiError(response.status, detailOf(parsed, `Erreur ${response.status}`), parsed, retryAfterOf(response));
    error.request = { path, method: init.method ?? "GET", body: init.body };
    // L'appelant reçoit toujours l'erreur (detail affiché) ; la popup de risques est ouverte en plus.
    if (error.status === 428 && error.source) consentListeners.forEach((listener) => listener(error));
    throw error;
  }
  return parsed as T;
}

/** URL d'un contenu HTML servi par l'API (ouvert dans un nouvel onglet, donc jeton en paramètre). */
export function authedUrl(path: string): string {
  return `/api${path}${token ? `?token=${encodeURIComponent(token)}` : ""}`;
}

export const enc = encodeURIComponent;

// ------------------------------------------------------------------ types
export type TaskStatus =
  | "en_attente" | "en_cours" | "attente_verification_manuelle" | "attente_saisie" | "terminée" | "échec" | "annulée";

export interface Task {
  id: string;
  kind: string;
  title: string;
  status: TaskStatus;
  logs: string[];
  result: Record<string, unknown> | null;
  error: string;
  needs_input: { question: string; field: string; options: string[]; suggestion: string; review: string } | null;
  meta: Record<string, unknown>;
  created_at: string;
  /** `null` tant que la tâche n'a pas démarré (contrat vague 1, C-2). */
  started_at: string | null;
  /** `null` tant que la tâche n'est pas finie (contrat vague 1, C-2). */
  finished_at: string | null;
}

export const FINAL_STATUSES: TaskStatus[] = ["terminée", "échec", "annulée"];

/** `result` d'une tâche `run` terminée (E-4 option A, G-10) : `blocked` si LinkedIn a interrompu la collecte. */
export interface RunResult {
  report: string;
  jobs: number;
  counts: Record<string, number>;
  /** Toujours présent (G-10). */
  blocked: boolean;
  /** Présent seulement si `blocked`. */
  blocked_where?: "collecte";
}

/** Vrai pour une analyse terminée dont la collecte a été interrompue par LinkedIn (rapport partiel). */
export function isBlockedRun(task: Task): boolean {
  return task.kind === "run" && task.status === "terminée" && (task.result as RunResult | null)?.blocked === true;
}

export interface Quota { used: number; limit: number }

export interface StatusInfo {
  session: { age_hours: number | null; ttl_hours: number; valid: boolean };
  credentials_configured: boolean;
  smtp_configured: boolean;
  anthropic_configured: boolean;
  interpreter: string;
  browser: string;
  headless: boolean;
  profile_file: boolean;
  has_analysis: boolean;
  quotas: { apply: Quota; messages: Quota };
  active_tasks: number;
  blocked: { at: string; where: string; until: string } | null;
}

export interface Settings {
  keyword: string;
  location: string;
  remote_only: boolean;
  caution: "prudent" | "normal" | "agressif";
  max_items: number | null;
  profile_source: "auto" | "file" | "linkedin";
  /** G-4 : `none` n'existe plus (lu comme `local` par le serveur). */
  interpreter: "claude" | "local";
  apply_min_score: number;
  apply_daily_max: number;
  message_daily_max: number;
  contact_email: string;
  contact_phone: string;
  signature: string;
  linkedin_skill_sync: boolean;
  answers: Record<string, string>;
}

export interface Job {
  id: string;
  title: string;
  company: string;
  url: string;
  location: string;
  posted_at: string;
  score: number;
  criteria: Record<string, number>;
  job_skills: string[];
  matched_skills: string[];
  missing_skills: string[];
  summary?: string[];
  years_required: number | null;
  seniority_required: string;
  remote: boolean;
  freelance: boolean;
  label: 0 | 1 | null;
  application_status: string | null;
  /** Extraits de l'en-tête LinkedIn des anciennes collectes (« il y a 1 jour », « Plus de 100 candidatures »). */
  posted_label?: string;
  applicants?: string;
}

export interface ModelInfo {
  trained: boolean;
  n_labels: number;
  train_accuracy: number | null;
  importance: { feature: string; weight: number }[];
  trained_at: string | null;
}

export interface JobsResponse {
  report: string | null;
  generated_at?: string;
  keyword?: string;
  model?: ModelInfo;
  min_labels?: number;
  jobs: Job[];
}

export interface JobDetail extends Job {
  text: string;
  owned_skills: string[];
}

export interface Contact {
  /** Contrat vague 1, C-4 : remplace `name` (lecture et écriture). */
  full_name: string;
  profile_url: string;
  email: string | null;
  headline?: string;
  /** Contrat vague 1, C-5 : origine du contact. */
  source: "linkedin_equipe_recrutement" | "manuel";
  /**
   * Vague 4 (backend/01 § 5.7.1), lecture seule : opposition enregistrée (I-13). Le serveur l'ignore en écriture ;
   * absent sur une ligne ajoutée localement.
   */
  opposed?: boolean;
  /** Vague 4 : date de collecte (UTC `Z`), pour la ligne « Origine de l'adresse » ; absente sur une ligne locale. */
  collected_at?: string | null;
}

/** Origine d'une adresse de destinataire (I-12). */
export type ContactOrigin = Contact["source"];

/** `GET /api/applications/{job_id}/message-draft?to=` (backend/01 § 5.7.1). */
export interface MessageDraft {
  subject: string;
  /** Corps sans le pied. */
  body: string;
  /** Pied imposé exact que le serveur ajoutera après `body` (origine + désinscription) : affiché, jamais modifiable. */
  footer: string;
  origin: ContactOrigin;
  /** Destinataire présent dans la liste d'opposition : envoi impossible (I-13). */
  opposed: boolean;
}

/** Motifs d'une opposition (`POST /api/suppressions`). */
export const SUPPRESSION_REASONS = ["désinscription", "refus_oral", "plainte"] as const;
export type SuppressionReason = (typeof SUPPRESSION_REASONS)[number];

/** Corps de `POST /api/suppressions` (201 `{"ok": true}`). */
export interface SuppressionBody { recipient: string; reason: SuppressionReason }

/** Détail exact du 409 d'envoi vers un destinataire opposé (backend/01 § 5.7.1). */
export const OPPOSED_DETAIL = "Ce destinataire a demandé à ne plus être contacté.";

export interface HistoryEntry { at: string; kind: string; [key: string]: unknown }

export interface ApplicationRow {
  id: string;
  title: string;
  company: string;
  url: string;
  score: number;
  status: string;
  generator: string;
  prepared_at: string;
  /** `null` si la candidature n'a jamais été modifiée (contrat vague 1, C-2). */
  updated_at: string | null;
  contacts: Contact[];
  history: HistoryEntry[];
  apply_state: string;
}

export interface ApplicationDetail extends ApplicationRow {
  cv: string;
  message: string;
  advice: string;
  offer: string;
  statuses: string[];
}

export interface Profile {
  available: boolean;
  message?: string;
  origin?: string;
  title?: string;
  skills: string[];
  base_skills: string[];
  years_experience?: number | null;
  seniority?: string;
  overrides: { title: string; years_experience: number | null; seniority: string; added_skills: string[]; removed_skills: string[] };
  suggestions: { missing: { skill: string; weight: number }[]; rising: { skill: string; growth: number }[] };
  scores_outdated: boolean;
}

/** Réponse de `POST /profile/skills` : la compétence est aussi ajoutée au profil LinkedIn (tâche de fond). */
export interface SkillAdded extends Profile {
  linkedin_sync: { status: "queued" | "skipped" | "unavailable" | "off"; message?: string; task?: Task };
}

export interface Monitor {
  generated_at: string;
  by_status: Record<string, number>;
  funnel: { step: string; count: number }[];
  quotas: { apply: Quota; messages: Quota };
  activity: ({ date: string } & Record<string, number | string>)[];
  recent_events: { at: string; kind: string; [key: string]: unknown }[];
  model: ModelInfo | null;
  last_analysis: { report: string; generated_at: string; keyword: string; counts: Record<string, number> } | null;
  active_tasks: Task[];
  recent_tasks: Task[];
  errors: { id: string; title: string; error: string; at: string }[];
}

export interface Report {
  generated_at: string;
  request: { keyword: string };
  profile: { skills: string[]; title: string; years_experience: number | null; seniority: string };
  counts: Record<string, number>;
  top_terms: { term: string; docs: number; share: number; words: number }[];
  skill_demand: { skill: string; total: number; job: number; person: number }[];
  rising: {
    recent_docs: number;
    older_docs: number;
    terms: { term: string; recent: number; older: number; growth: number }[];
    skills: { term: string; recent: number; older: number; growth: number }[];
  };
  weekly: { weeks: string[]; counts: number[]; series: Record<string, number[]> };
  gap: { coverage: number; missing: { skill: string; weight: number }[]; skills: { skill: string; weight: number; owned: boolean }[] };
  interpretation: string;
  interpreter: string;
  model: ModelInfo;
  /** Offres classées, avec annotation et statut de candidature. */
  top_jobs: Job[];
  /** Les actions sur une offre (détail, préparation) visent le dernier rapport uniquement. */
  is_latest: boolean;
}

export interface ReportRow { name: string; keyword: string; generated_at: string; jobs: number; html: boolean }

// ------------------------------------------------------------------ sources et consentements (I-16, vague 3)
export type SourceCode = "linkedin" | "france_travail" | "malt" | "wttj" | "indeed";
export type AccessMode = "navigateur" | "api_officielle" | "import_manuel";

/** Élément de `GET /api/sources` (backend/01 § 5.8 et § 5.8.1). */
export interface SourceInfo {
  code: SourceCode;
  label: string;
  access_mode: AccessMode;
  /** `false` : source pas encore disponible dans cette version. */
  enabled: boolean;
  risk_version: string | null;
  consent: { risk_version: string; accepted_at: string } | null;
  /** Mode `navigateur` sans consentement actif à la version courante. */
  consent_required: boolean;
  /** France Travail : identifiants présents dans le .env. */
  configured?: boolean;
  /** Repos de 6 h propre à la source. */
  blocked: { at: string; where: string; until: string } | null;
}

/** Corps de `POST /api/sources/{code}/consent`. */
export interface ConsentBody { risk_version: string; accepted: true }
