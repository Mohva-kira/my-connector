import {
  QueryClient, QueryKey, UseQueryOptions, keepPreviousData, useMutation, useQuery, useQueryClient,
} from "@tanstack/react-query";
import { useEffect, useState, useSyncExternalStore } from "react";
import {
  ApplicationDetail, ApplicationRow, FINAL_STATUSES, JobDetail, JobsResponse, ManualItemBody, ManualItemCreated, Monitor, Profile, Report, ReportRow, Settings,
  SourceCode, SourceInfo, StatusInfo, Task, api, enc,
} from "./api";
import { reportSchema } from "./schemas";

// ------------------------------------------------------------------ jeton périmé (401)
// Après un 401, le jeton local a changé (redémarrage de l'application) : toute requête échouerait. L'interrogation
// périodique et les requêtes sont suspendues jusqu'au rechargement de la page.
let tokenExpired = false;
const expiryListeners = new Set<() => void>();

/** Signale un 401 : suspend toutes les requêtes et l'interrogation périodique. */
export function markTokenExpired() {
  if (tokenExpired) return;
  tokenExpired = true;
  expiryListeners.forEach((listener) => listener());
}

const subscribeExpiry = (listener: () => void) => {
  expiryListeners.add(listener);
  return () => { expiryListeners.delete(listener); };
};

/** Vrai après un 401, jusqu'au rechargement. */
export const useTokenExpired = () => useSyncExternalStore(subscribeExpiry, () => tokenExpired);

type LiveQueryOptions<T> = Omit<UseQueryOptions<T, Error, T, QueryKey>, "enabled"> & { enabled?: boolean };

/** `useQuery` suspendu après un 401 (requête désactivée, plus d'interrogation périodique). */
function useLiveQuery<T>({ enabled = true, refetchInterval, ...options }: LiveQueryOptions<T>) {
  const live = !useTokenExpired();
  return useQuery({ ...options, enabled: live && enabled, refetchInterval: live ? refetchInterval : false });
}

export const useStatus = () => useLiveQuery({ queryKey: ["status"], queryFn: () => api<StatusInfo>("/status"), refetchInterval: 15000 });
export const useMonitor = () => useLiveQuery({ queryKey: ["monitor"], queryFn: () => api<Monitor>("/monitor"), refetchInterval: 10000 });
export const useSettings = () => useLiveQuery({ queryKey: ["settings"], queryFn: () => api<Settings>("/settings") });
export const useProfile = () => useLiveQuery({ queryKey: ["profile"], queryFn: () => api<Profile>("/profile") });
export const useApplications = () => useLiveQuery({ queryKey: ["applications"], queryFn: () => api<ApplicationRow[]>("/applications") });
/** Sources et consentements (I-16). Invalidée après acceptation, révocation ou levée d'un blocage. */
export const useSources = () => useLiveQuery({ queryKey: ["sources"], queryFn: () => api<SourceInfo[]>("/sources") });
export const useReports = () => useLiveQuery({ queryKey: ["reports"], queryFn: () => api<ReportRow[]>("/reports") });

/** Rapport validé par Zod : les sections absentes d'un rapport ancien retombent sur des valeurs vides. */
async function fetchReport(name: string): Promise<Report> {
  const parsed = reportSchema.safeParse(await api<unknown>(`/reports/${enc(name)}`));
  if (!parsed.success) throw new Error("Rapport illisible : son contenu ne correspond pas au format attendu.");
  return parsed.data;
}

export function useReport(name?: string) {
  return useLiveQuery({ queryKey: ["report", name], queryFn: () => fetchReport(name!), enabled: !!name });
}

/** Valeur recopiée après `delay` ms sans changement (anti-rebond). */
export function useDebouncedValue<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

/** Offres filtrées : recherche `q` avec anti-rebond de 300 ms, liste précédente gardée pendant le chargement. */
export function useJobs(filters: Record<string, string>) {
  const q = useDebouncedValue(filters.q ?? "", 300);
  const effective = { ...filters, ...(filters.q !== undefined ? { q } : {}) };
  const query = new URLSearchParams(Object.entries(effective).filter(([, v]) => v !== "")).toString();
  return useLiveQuery({
    queryKey: ["jobs", query],
    queryFn: () => api<JobsResponse>(`/jobs${query ? `?${query}` : ""}`),
    placeholderData: keepPreviousData,
  });
}

export const useJob = (id?: string) =>
  useLiveQuery({ queryKey: ["job", id], queryFn: () => api<JobDetail>(`/jobs/${enc(id!)}`), enabled: !!id });

export const useApplication = (id?: string) =>
  useLiveQuery({ queryKey: ["application", id], queryFn: () => api<ApplicationDetail>(`/applications/${enc(id!)}`), enabled: !!id });

const isFinal = (task: Task) => FINAL_STATUSES.includes(task.status);

/** Tâches récentes : interrogées toutes les 2 s tant qu'une tâche est active, toutes les 15 s sinon. */
export const useTasks = () =>
  useLiveQuery({
    queryKey: ["tasks"],
    queryFn: () => api<Task[]>("/tasks"),
    refetchInterval: (query) => (query.state.data?.some((task) => !isFinal(task)) ? 2000 : 15000),
  });

/** Détail d'une tâche : 1,5 s tant que son statut n'est pas final (lot 1, 02). */
export const useTask = (id?: string) =>
  useLiveQuery({
    queryKey: ["task", id],
    queryFn: () => api<Task>(`/tasks/${id}`),
    enabled: !!id,
    refetchInterval: (query) => (query.state.data && isFinal(query.state.data) ? false : 1500),
  });

/**
 * Clés périmées à la fin d'une tâche, selon son type (frontend/03 § 2.3). `sources` : une tâche pilotant un navigateur
 * peut avoir posé un repos de 6 h (I-06) que le bandeau de blocage lit dans `GET /api/sources`. `reports` / `report` :
 * `run`, `rescore` et `train` écrivent un nouveau rapport (le précédent perd son statut de dernier rapport).
 */
const INVALIDATE_BY_KIND: Record<string, string[][]> = {
  run: [["reports"], ["report"], ["jobs"], ["monitor"], ["profile"], ["status"], ["sources"]],
  rescore: [["reports"], ["report"], ["jobs"], ["monitor"], ["profile"]],
  train: [["reports"], ["report"], ["jobs"], ["monitor"], ["profile"]],
  prepare: [["applications"], ["application"], ["jobs"], ["monitor"], ["status"], ["sources"]],
  apply: [["applications"], ["application"], ["jobs"], ["monitor"], ["status"], ["sources"]],
  contacts: [["applications"], ["application"], ["jobs"], ["monitor"], ["status"], ["sources"]],
  message: [["applications"], ["application"], ["jobs"], ["monitor"], ["status"], ["sources"]],
  import_profile: [["profile"], ["status"], ["sources"]],
  profile_skill: [["profile"], ["status"], ["sources"]],
  // Purge de rétention (I-10) : offres supprimées (liste, détail) et compteurs du tableau de bord.
  purge: [["jobs"], ["job"], ["monitor"]],
};

/** Invalide le cache après la fin d'une tâche ; type inconnu ⇒ tout le cache (sûr par défaut). */
export function invalidateAfterTask(client: QueryClient, task: Task) {
  const keys = INVALIDATE_BY_KIND[task.kind];
  if (!keys) {
    client.invalidateQueries();
    return;
  }
  [["tasks"], ["task", task.id], ...keys].forEach((queryKey) => client.invalidateQueries({ queryKey }));
}

/** Mutation qui invalide les requêtes listées après succès. */
export function useAction<TVars, TResult>(fn: (vars: TVars) => Promise<TResult>, invalidate: string[][] = []) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => invalidate.forEach((key) => client.invalidateQueries({ queryKey: key })),
  });
}

/**
 * Annotation d'une offre (`PUT /jobs/{id}/label`). Invalide aussi `["report"]` : Marché affiche les annotations du
 * rapport et ne doit pas rester périmé après une annotation faite depuis Offres (frontend/03 § 2.3).
 */
export const useLabelJob = () =>
  useAction(
    ({ id, value }: { id: string; value: 0 | 1 | null }) => api(`/jobs/${enc(id)}/label`, { method: "PUT", body: { value } }),
    [["jobs"], ["report"], ["monitor"]]);

const SOURCE_KEYS = [["sources"], ["status"]];

/** Acceptation des risques d'une source (`POST /api/sources/{code}/consent`, 201). */
export const useAcceptConsent = () =>
  useAction(
    ({ code, risk_version }: { code: SourceCode; risk_version: string }) =>
      api<SourceInfo>(`/sources/${enc(code)}/consent`, { method: "POST", body: { risk_version, accepted: true } }),
    SOURCE_KEYS);

/** Révocation (`DELETE /api/sources/{code}/consent`, 204) : la prochaine action réaffiche la popup. */
export const useRevokeConsent = () =>
  useAction((code: SourceCode) => api<null>(`/sources/${enc(code)}/consent`, { method: "DELETE" }), SOURCE_KEYS);

/** Levée du repos d'une source (`POST /api/block/clear {source}`). */
export const useClearBlock = () =>
  useAction((source: SourceCode) => api<{ ok: boolean }>("/block/clear", { method: "POST", body: { source } }), SOURCE_KEYS);

/**
 * Import manuel d'une offre Indeed (`POST /api/items/import`, 201). Invalide les offres, le rapport et le suivi ; pas
 * `["tasks"]` : aucune tâche n'est créée (l'offre est scorée au prochain « Recalculer les scores »).
 */
export const useImportItem = () =>
  useAction((body: ManualItemBody) => api<ManualItemCreated>("/items/import", { method: "POST", body }), [["jobs"], ["report"], ["monitor"]]);
