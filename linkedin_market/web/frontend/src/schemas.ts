// Schémas Zod : bornes et messages identiques au serveur (backend/01 § 1.2 et § 5.2). La validation client n'est
// qu'un confort : le serveur reste seul garant.
import { z } from "zod";
import { SUPPRESSION_REASONS } from "./api";
import type { Job, ManualItemBody, ModelInfo, Report, Settings, SuppressionBody } from "./api";

// ------------------------------------------------------------------ prudence (I-07, E-3 / G-3)
export const CAUTIONS = ["prudent", "normal", "agressif"] as const;
export type Caution = (typeof CAUTIONS)[number];
/** Plafond d'éléments par type selon la prudence (`Caution.max_items`). */
export const CAUTION_MAX: Record<Caution, number> = { prudent: 15, normal: 30, agressif: 60 };
/** Prudence par défaut décidée (E-3). */
export const DEFAULT_CAUTION: Caution = "prudent";

/** Message serveur exact (G-3). */
export const maxItemsCapMessage = (caution: Caution) =>
  `max_items ne peut pas dépasser ${CAUTION_MAX[caution]} en prudence « ${caution} ».`;

/** Même expression que `outreach._EMAIL` côté serveur (G-12). */
export const EMAIL_RE = /^[^@\s,;<>]+@[^@\s,;<>]+\.[^@\s,;<>]+$/;

const cautionSchema = z.enum(CAUTIONS, { error: "caution doit être l'un de : prudent, normal, agressif." });
const interpreterSchema = z.enum(["claude", "local"], { error: "interpreter doit être l'un de : claude, local." });
const maxItemsSchema = z
  .number({ error: "max_items doit être un entier." })
  .int({ error: "max_items doit être un entier." })
  .min(1, { error: "max_items doit être au moins 1." })
  .nullable();

/** `max_items` ≤ plafond de la prudence choisie (même message que le serveur). */
function checkMaxItems(value: { caution: Caution; max_items?: number | null }, ctx: z.RefinementCtx) {
  if (value.max_items != null && value.max_items > CAUTION_MAX[value.caution]) {
    ctx.addIssue({ code: "custom", path: ["max_items"], message: maxItemsCapMessage(value.caution) });
  }
}

const entier = (field: string, min: number, max: number) =>
  z.number({ error: `${field} doit être compris entre ${min} et ${max}.` })
    .int({ error: `${field} doit être compris entre ${min} et ${max}.` })
    .min(min, { error: `${field} doit être compris entre ${min} et ${max}.` })
    .max(max, { error: `${field} doit être compris entre ${min} et ${max}.` });

/** Vide accepté ; sinon le nom doit être connu d'Intl (confort : le serveur reste juge). */
function isValidTimezone(v: string): boolean {
  if (v === "") return true;
  try {
    new Intl.DateTimeFormat("fr-FR", { timeZone: v });
    return true;
  } catch {
    return false;
  }
}

// ------------------------------------------------------------------ réglages (PUT /api/settings)
export const settingsSchema = z.object({
  keyword: z.string().trim(),
  location: z.string().trim(),
  home_location: z.string().trim()
    .refine((v) => v.replace(/\s+/g, " ").length <= 80, { error: "home_location ne peut pas dépasser 80 caractères." })
    // eslint-disable-next-line no-control-regex
    .refine((v) => !/[\u0000-\u001f\u007f-\u009f]/.test(v), { error: "home_location ne doit contenir ni saut de ligne ni caractère de contrôle." }),
  timezone: z.string().trim().refine(isValidTimezone, { error: "timezone doit être un fuseau horaire IANA valide (ex. Europe/Paris) ou vide." }),
  remote_only: z.boolean(),
  caution: cautionSchema,
  max_items: maxItemsSchema,
  profile_source: z.enum(["auto", "file", "linkedin"], { error: "profile_source doit être l'un de : auto, file, linkedin." }),
  interpreter: interpreterSchema,
  apply_min_score: z.number({ error: "apply_min_score doit être un nombre." })
    .min(0, { error: "apply_min_score doit être compris entre 0 et 100." })
    .max(100, { error: "apply_min_score doit être compris entre 0 et 100." }),
  apply_daily_max: entier("apply_daily_max", 0, 200),
  message_daily_max: entier("message_daily_max", 0, 200),
  // Vide admis ; sinon adresse valide (G-12, message serveur exact).
  contact_email: z.string().trim().refine((v) => v === "" || EMAIL_RE.test(v), { error: "contact_email doit être une adresse email valide." }),
  contact_phone: z.string().trim(),
  signature: z.string().trim(),
  linkedin_skill_sync: z.boolean(),
  answers: z.record(z.string(), z.string(), { error: "answers doit être un objet question -> réponse." }),
}).superRefine(checkMaxItems) satisfies z.ZodType<Settings>;

// ------------------------------------------------------------------ lancement d'analyse (POST /api/runs)
export const runBodySchema = z.object({
  scrape: z.boolean(),
  keyword: z.string().trim().min(1, { error: "Renseigne le mot-clé du poste visé (Paramètres ou lancement d'analyse)." }),
  location: z.string().trim(),
  remote_only: z.boolean(),
  caution: cautionSchema.default(DEFAULT_CAUTION),
  max_items: maxItemsSchema.optional(),
  // Ignoré par le serveur si scrape = false ; défaut serveur ["linkedin"].
  sources: z.array(z.string()).optional(),
}).superRefine((value, ctx) => {
  checkMaxItems(value, ctx);
  if (value.scrape && value.sources && value.sources.length === 0) {
    ctx.addIssue({ code: "custom", path: ["sources"], message: "Choisis au moins une source." });
  }
});
export type RunBody = z.infer<typeof runBodySchema>;

// ------------------------------------------------------------------ opposition (POST /api/suppressions, I-13)
/** Même expression que la validation des contacts côté serveur (`Contact.profile_url`). */
export const LINKEDIN_PROFILE_RE = /^https:\/\/([a-z]{2,3}\.)?linkedin\.com\/in\/[^/?#\s]+/;

export const suppressionSchema = z.object({
  // Message serveur exact (backend/01 § 5.7.1).
  recipient: z.string().trim().refine((v) => EMAIL_RE.test(v) || LINKEDIN_PROFILE_RE.test(v), {
    error: "recipient : adresse email ou URL de profil LinkedIn attendue.",
  }),
  reason: z.enum(SUPPRESSION_REASONS, { error: "reason : doit être l'un de : désinscription, refus_oral, plainte" }),
}) satisfies z.ZodType<SuppressionBody>;

// ------------------------------------------------------------------ import manuel Indeed (POST /api/items/import, R-6)
/** Hôte `indeed.com` ou sous-domaine, en https (backend/01 § 5.8.2) : `evil-indeed.com` et `indeed.com.evil.fr` refusés. */
export const INDEED_HOST_RE = /^https:\/\/([a-z0-9-]+\.)*indeed\.com(\/|$|\?|#)/i;
export const MANUAL_TEXT_MIN = 200;
export const MANUAL_TEXT_MAX = 20000;

/** Bornes et messages du serveur ; `trim` avant `min`/`max`. `source_code` est ajouté à l'envoi. */
export const manualImportSchema = z.object({
  url: z.string().trim().max(2000, { error: "colle l'adresse https de l'offre Indeed." })
    .refine((v) => INDEED_HOST_RE.test(v), { error: "colle l'adresse https de l'offre Indeed." }),
  title: z.string().trim()
    .min(1, { error: "renseigne le titre de l'offre (150 caractères maximum)." })
    .max(150, { error: "renseigne le titre de l'offre (150 caractères maximum)." }),
  company: z.string().trim().max(80, { error: "entreprise trop longue (80 caractères maximum)." }),
  location: z.string().trim().max(120, { error: "lieu trop long (120 caractères maximum)." }),
  text: z.string().trim()
    .min(MANUAL_TEXT_MIN, { error: "colle le texte complet de l'offre (200 caractères minimum)." })
    .max(MANUAL_TEXT_MAX, { error: "texte trop long (20 000 caractères maximum)." }),
  keyword: z.string().trim().max(80, { error: "mot-clé trop long (80 caractères maximum)." }),
}) satisfies z.ZodType<Omit<ManualItemBody, "source_code" | "keyword"> & { keyword: string }>;
export type ManualImportForm = z.infer<typeof manualImportSchema>;

/** Premier message d'erreur par champ (clé = premier élément du chemin, `""` pour l'objet). */
export function fieldErrors(error: z.ZodError): Record<string, string> {
  const out: Record<string, string> = {};
  for (const issue of error.issues) {
    const key = String(issue.path[0] ?? "");
    out[key] ??= issue.message;
  }
  return out;
}

// ------------------------------------------------------------------ rapport (garde de Marché)
// Un rapport ancien ou importé peut manquer de sections : chaque partie retombe sur une valeur vide au lieu de faire
// planter la page ; l'interface affiche alors « Données insuffisantes ».
const str = z.string().catch("");
const num = z.number().catch(0);
const strList = z.array(z.string()).catch([]);

const trendSchema = z.object({ term: z.string(), recent: num, older: num, growth: num });

const modelSchema = z.object({
  trained: z.boolean().catch(false),
  n_labels: num,
  train_accuracy: z.number().nullable().catch(null),
  importance: z.array(z.object({ feature: z.string(), weight: num })).catch([]),
  trained_at: z.string().nullable().catch(null),
}) satisfies z.ZodType<ModelInfo>;

const EMPTY_MODEL: ModelInfo = { trained: false, n_labels: 0, train_accuracy: null, importance: [], trained_at: null };

const jobSchema = z.looseObject({
  id: z.string(),
  // Rapport antérieur à la vague 5 (ou valeur inconnue) : lu comme LinkedIn, rien n'est réécrit (I-02).
  source_code: z.enum(["linkedin", "france_travail", "malt", "wttj", "indeed"]).catch("linkedin"),
  title: str,
  company: str,
  url: str,
  location: str,
  posted_at: str,
  score: num,
  criteria: z.record(z.string(), z.number()).catch({}),
  job_skills: strList,
  matched_skills: strList,
  missing_skills: strList,
  summary: z.array(z.string()).optional().catch(undefined),
  years_required: z.number().nullable().catch(null),
  seniority_required: str,
  remote: z.boolean().catch(false),
  freelance: z.boolean().catch(false),
  label: z.union([z.literal(0), z.literal(1), z.null()]).catch(null),
  application_status: z.string().nullable().catch(null),
  posted_label: z.string().optional().catch(undefined),
  applicants: z.string().optional().catch(undefined),
}) satisfies z.ZodType<Job>;

export const reportSchema = z.object({
  generated_at: str,
  request: z.looseObject({ keyword: str }).catch({ keyword: "" }),
  profile: z.object({
    skills: strList,
    title: str,
    years_experience: z.number().nullable().catch(null),
    seniority: str,
  }).catch({ skills: [], title: "", years_experience: null, seniority: "" }),
  counts: z.record(z.string(), z.number()).catch({}),
  top_terms: z.array(z.object({ term: z.string(), docs: num, share: num, words: num })).catch([]),
  skill_demand: z.array(z.object({ skill: z.string(), total: num, job: num, person: num })).catch([]),
  rising: z.object({
    recent_docs: num,
    older_docs: num,
    terms: z.array(trendSchema).catch([]),
    skills: z.array(trendSchema).catch([]),
  }).catch({ recent_docs: 0, older_docs: 0, terms: [], skills: [] }),
  weekly: z.object({
    weeks: strList,
    counts: z.array(z.number()).catch([]),
    series: z.record(z.string(), z.array(z.number())).catch({}),
  }).catch({ weeks: [], counts: [], series: {} }),
  gap: z.object({
    coverage: num,
    missing: z.array(z.object({ skill: z.string(), weight: num })).catch([]),
    skills: z.array(z.object({ skill: z.string(), weight: num, owned: z.boolean().catch(false) })).catch([]),
  }).catch({ coverage: 0, missing: [], skills: [] }),
  interpretation: str,
  interpreter: str,
  model: modelSchema.catch(EMPTY_MODEL),
  top_jobs: z.array(jobSchema).catch([]),
  is_latest: z.boolean().catch(false),
}) satisfies z.ZodType<Report>;
