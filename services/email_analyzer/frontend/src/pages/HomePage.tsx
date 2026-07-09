import React, { useCallback, useEffect, useState } from "react";
import { Link, Navigate } from "react-router-dom";
import AnalysisDashboard, {
  type DraftResult,
  type ProjectAnalysis,
} from "../AnalysisDashboard";
import ConversationalAssistant, { type ChatTurn } from "../ConversationalAssistant";
import { apiFetch, getAccessToken } from "../apiClient";
import { parseApiError, parseResponseJson, type Health } from "../apiUtils";
import AppShell from "../components/AppShell";
import { useSaasSession } from "../SaasPanels";
import { applyAnalyzeTickToStore, type AnalyzeJobTick } from "../sync-experience/bridgeAnalyzeJob";
import { useSyncStore } from "../store/syncStore";

type AssistantProvider = "openai" | "gemini" | "none";

// Polling du job d'analyse : /api/analyze renvoie un job_id (202), on interroge
// /api/analyze/{job_id} jusqu'à done/error. Garde chaque requête HTTP courte pour
// éviter les timeouts de passerelle (Nginx / zrok).
const ANALYZE_POLL_INTERVAL_MS = 2500;
const ANALYZE_POLL_TIMEOUT_MS = 5 * 60 * 1000;

// Messages courts qui tournent pendant le chargement (gamification sobre :
// donne un retour vivant sans bruit visuel — cf. context/ui-context.md).
const LOADING_STATUS_MESSAGES = [
  "Récupération des emails…",
  "Analyse en cours…",
  "Repérage des emails critiques…",
];

type AnalyzeProgress = { processed: number; total: number };

// Taille de lot affichée dans la pastille de gamification (doit correspondre à
// EMAIL_ANALYZER_BATCH_SIZE côté serveur, défaut 10 — purement décoratif, ne
// pilote aucune logique).
const BATCH_CHUNK_SIZE_HINT = 10;

async function pollAnalysis(
  jobId: string,
  onTick: (data: Record<string, unknown>) => void,
): Promise<Record<string, unknown>> {
  const deadline = Date.now() + ANALYZE_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, ANALYZE_POLL_INTERVAL_MS));
    // Timeout par requête : un poll ne doit jamais pendre indéfiniment. Un
    // tick isolé qui échoue (abort 30s, hoquet réseau) ne doit pas tuer tout
    // le run tant que le budget global (deadline) n'est pas dépassé — le job
    // backend continue de tourner et le prochain tick peut réussir.
    let data: Record<string, unknown>;
    try {
      const res = await apiFetch(`/api/analyze/${jobId}`, {
        method: "GET",
        signal: AbortSignal.timeout(30_000),
      });
      data = (await parseResponseJson(res)) as Record<string, unknown>;
      if (!res.ok) {
        throw new Error(parseApiError(data));
      }
    } catch {
      continue;
    }
    onTick(data);
    const status = data.status as string;
    if (status === "done") {
      return (data.result ?? {}) as Record<string, unknown>;
    }
    if (status === "error") {
      throw new Error((data.error as string) || "Échec de l'analyse.");
    }
  }
  throw new Error("L'analyse prend trop de temps. Réessayez plus tard.");
}

export default function HomePage({
  health,
  healthError,
  saasEnabled,
  billingReturnBanner,
  sessionTick,
}: {
  health: Health | null;
  healthError: string | null;
  saasEnabled: boolean;
  billingReturnBanner: boolean;
  sessionTick: number;
}) {
  const [project, setProject] = useState("");
  const [periodMode, setPeriodMode] = useState<"window" | "preset">("window");
  const [period, setPeriod] = useState<string>("7");
  const [days, setDays] = useState(30);
  const [provider, setProvider] = useState<AssistantProvider>("openai");
  const [openaiModel, setOpenaiModel] = useState("gpt-4o-mini");
  const [geminiModel, setGeminiModel] = useState("gemini-2.5-flash");
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [activeProjectName, setActiveProjectName] = useState<string | null>(null);
  const [analysisBlock, setAnalysisBlock] = useState<ProjectAnalysis | null>(null);
  const [runSuccess, setRunSuccess] = useState(false);
  const [resultPanelKey, setResultPanelKey] = useState(0);
  const [progress, setProgress] = useState<AnalyzeProgress | null>(null);
  const [partialBlock, setPartialBlock] = useState<ProjectAnalysis | null>(null);
  const [statusMsgIndex, setStatusMsgIndex] = useState(0);
  const [responseMode, setResponseMode] = useState<"classic" | "conversational">("classic");
  const [chatMessages, setChatMessages] = useState<ChatTurn[]>([]);

  const { me } = useSaasSession(saasEnabled, sessionTick);

  useEffect(() => {
    setChatMessages([]);
    setResponseMode("classic");
  }, [activeProjectName, resultPanelKey]);

  useEffect(() => {
    if (!loading) return;
    const id = setInterval(() => {
      setStatusMsgIndex((i) => (i + 1) % LOADING_STATUS_MESSAGES.length);
    }, 2200);
    return () => clearInterval(id);
  }, [loading]);

  const pickProjectBlock = useCallback(
    (data: Record<string, unknown>, projectFilter: string) => {
      const keys = Object.keys(data).filter((k) => !k.startsWith("_"));
      if (keys.length === 0) return { name: null as string | null, block: null as ProjectAnalysis | null };
      const key =
        keys.find((k) => k.toLowerCase() === projectFilter.trim().toLowerCase()) ||
        keys[0];
      const block = data[key];
      if (block && typeof block === "object") {
        return { name: key, block: block as ProjectAnalysis };
      }
      return { name: null, block: null };
    },
    [],
  );

  async function runAnalyze(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setReport(null);
    setAnalysisBlock(null);
    setActiveProjectName(null);
    setRunSuccess(false);
    setProgress(null);
    setPartialBlock(null);
    setStatusMsgIndex(0);
    setLoading(true);
    useSyncStore.getState().reset();
    try {
      const body = {
        project: project.trim(),
        period: periodMode === "preset" && period ? period : null,
        days: periodMode === "window" ? days : 30,
        assistant_provider: provider,
        openai_model: openaiModel,
        gemini_model: geminiModel,
      };
      const startRes = await apiFetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const startData = (await parseResponseJson(startRes)) as Record<string, unknown>;
      if (!startRes.ok) {
        throw new Error(parseApiError(startData));
      }
      const jobId = startData.job_id as string | undefined;
      if (!jobId) {
        throw new Error("Réponse inattendue du serveur (job_id manquant).");
      }
      const data = await pollAnalysis(jobId, (tick) => {
        applyAnalyzeTickToStore(tick as unknown as AnalyzeJobTick, useSyncStore.getState());
        const tickProgress = tick.progress as Partial<AnalyzeProgress> | undefined;
        if (tickProgress && typeof tickProgress.total === "number" && tickProgress.total > 0) {
          setProgress({
            processed: tickProgress.processed ?? 0,
            total: tickProgress.total,
          });
        }
        const partial = tick.partial as Record<string, unknown> | null | undefined;
        if (partial && typeof partial === "object") {
          const { block } = pickProjectBlock(partial, project);
          if (block) setPartialBlock(block);
        }
      });
      setReport(data);
      const { name, block } = pickProjectBlock(data, project);
      setActiveProjectName(name);
      setAnalysisBlock(block);
      const hasKeys = Object.keys(data).filter((k) => !k.startsWith("_")).length > 0;
      setRunSuccess(hasKeys && block !== null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
      useSyncStore.getState().reset();
    } finally {
      setLoading(false);
      setResultPanelKey((k) => k + 1);
    }
  }

  const requestDraft = useCallback(async (): Promise<DraftResult> => {
    if (!analysisBlock || !activeProjectName) {
      throw new Error("Analyse manquante");
    }
    const res = await apiFetch("/api/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_name: activeProjectName,
        analysis: analysisBlock,
      }),
    });
    const data = await parseResponseJson(res);
    if (!res.ok) {
      throw new Error(parseApiError(data));
    }
    return data as DraftResult;
  }, [analysisBlock, activeProjectName]);

  const reportKeys = report
    ? Object.keys(report).filter((k) => !k.startsWith("_"))
    : [];
  const isEmptyReport =
    report !== null &&
    (reportKeys.length === 0 || report._empty === true);

  const showResultPanel = loading || report !== null || error !== null;
  const showPlaceholder = !loading && report === null && !error;

  if (saasEnabled && getAccessToken() && !me) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-bg-primary px-4">
        <p className="text-sm text-text-muted">Chargement du compte…</p>
      </div>
    );
  }

  if (saasEnabled && !getAccessToken()) {
    return <Navigate to="/login" replace />;
  }

  const inner = (
    <>
      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Analyse d&apos;emails</h1>
        <p className="mt-1 text-sm text-text-muted">
          {saasEnabled ? (
            <>
              Mode SaaS — boîte IMAP et quotas par organisation. Configurez la boîte dans{" "}
              <Link to="/settings" className="font-medium text-text-secondary underline decoration-border-default underline-offset-2 hover:text-text-primary">
                Paramètres
              </Link>
              .
            </>
          ) : (
            <>
              Lancez une analyse depuis votre boîte (identifiants IMAP dans le fichier{" "}
              <code className="rounded bg-bg-tertiary px-1 text-xs">.env</code> du dépôt).
            </>
          )}
        </p>
      </header>

      {billingReturnBanner ? (
        <div
          className="mb-6 rounded-2xl border border-success/30 bg-success/10 px-4 py-3 text-center text-sm text-success"
          role="status"
        >
          Retour depuis le paiement : si le montant est validé, votre organisation sera activée sous peu
          (notification serveur CinetPay).
        </div>
      ) : null}

      {healthError ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          Impossible de joindre l&apos;API ({healthError}). Vérifiez que{" "}
          <code className="text-xs">uvicorn</code> tourne sur le port 8000.
        </div>
      ) : null}

      {!saasEnabled && health && !health.imap_configured ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning"
        >
          IMAP non configuré : renseignez <code className="font-mono text-xs">IMAP_USER</code> et{" "}
          <code className="font-mono text-xs">IMAP_PASSWORD</code> dans{" "}
          <code className="font-mono text-xs">.env</code> à la racine du projet.
        </div>
      ) : null}

      {saasEnabled && me ? (
        <div
          role="status"
          className="mb-6 rounded-2xl border border-info/30 bg-info/10 px-4 py-3 text-sm text-info"
        >
          Organisation <strong>{me.tenants.find((t) => t.id === me.active_tenant_id)?.name ?? ""}</strong> — statut :{" "}
          <strong>{me.tenants.find((t) => t.id === me.active_tenant_id)?.status ?? ""}</strong>. Configurez la boîte
          IMAP dans{" "}
          <Link to="/settings" className="font-medium text-info underline underline-offset-2">
            Paramètres
          </Link>{" "}
          si ce n&apos;est pas encore fait.
        </div>
      ) : null}

      <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-10 xl:grid-cols-[380px_minmax(0,1fr)]">
        <div className="lg:max-w-md">
          <form
            onSubmit={runAnalyze}
            className="rounded-2xl border border-border-default bg-surface p-6 shadow-sm"
          >
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-text-secondary">Projet (filtre)</span>
              <input
                type="text"
                value={project}
                onChange={(e) => setProject(e.target.value)}
                required
                placeholder="Ex. BBCI, Nom du client…"
                className="mt-1 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2.5 text-sm text-text-primary outline-none ring-accent-primary transition placeholder:text-text-muted focus:border-accent-primary focus:bg-bg-tertiary focus:ring-2"
              />
            </label>

            <div className="mt-5">
              <span className="text-sm font-medium text-text-secondary">Période</span>
              <div className="mt-2 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setPeriodMode("window")}
                  className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
                    periodMode === "window"
                      ? "bg-accent-primary text-white"
                      : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
                  }`}
                >
                  Fenêtre (jours)
                </button>
                <button
                  type="button"
                  onClick={() => setPeriodMode("preset")}
                  className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
                    periodMode === "preset"
                      ? "bg-accent-primary text-white"
                      : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
                  }`}
                >
                  Préréglage
                </button>
              </div>
              {periodMode === "window" ? (
                <label className="mt-3 block">
                  <span className="text-xs text-text-muted">Nombre de jours (IMAP)</span>
                  <input
                    type="number"
                    min={1}
                    max={365}
                    value={days}
                    onChange={(e) => setDays(Number(e.target.value))}
                    className="mt-1 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2 text-sm text-text-primary outline-none focus:ring-2 focus:ring-accent-primary"
                  />
                </label>
              ) : (
                <select
                  value={period}
                  onChange={(e) => setPeriod(e.target.value)}
                  className="mt-3 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2.5 text-sm text-text-primary outline-none focus:ring-2 focus:ring-accent-primary"
                >
                  <option value="today">Aujourd&apos;hui</option>
                  <option value="yesterday">Hier</option>
                  <option value="3">3 derniers jours</option>
                  <option value="7">7 derniers jours</option>
                  <option value="11">11 derniers jours</option>
                </select>
              )}
            </div>

            <div className="mt-5">
              <span className="text-sm font-medium text-text-secondary">Assistant LLM</span>
              <div className="mt-2 flex flex-wrap gap-2">
                {(
                  [
                    ["openai", "OpenAI"],
                    ["gemini", "Gemini"],
                    ["none", "Sans LLM"],
                  ] as const
                ).map(([p, label]) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setProvider(p)}
                    className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
                      provider === p ? "bg-accent-primary text-white" : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>

            <button
              type="button"
              onClick={() => setAdvancedOpen(!advancedOpen)}
              className="mt-4 text-sm text-text-muted underline decoration-border-default underline-offset-2 hover:text-text-secondary"
            >
              {advancedOpen ? "Masquer" : "Avancé"} — modèles
            </button>
            {advancedOpen ? (
              <div className="mt-3 space-y-3 rounded-xl border border-border-subtle bg-bg-tertiary/40 p-3">
                <label className="block text-xs">
                  <span className="text-text-secondary">Modèle OpenAI</span>
                  <input
                    type="text"
                    value={openaiModel}
                    onChange={(e) => setOpenaiModel(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-1.5 text-sm text-text-primary"
                  />
                </label>
                <label className="block text-xs">
                  <span className="text-text-secondary">Modèle Gemini</span>
                  <input
                    type="text"
                    value={geminiModel}
                    onChange={(e) => setGeminiModel(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-1.5 text-sm text-text-primary"
                  />
                </label>
              </div>
            ) : null}

            <button
              type="submit"
              disabled={loading || !project.trim()}
              className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-accent-primary py-3 text-sm font-semibold text-white shadow-md shadow-black/20 transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span
                    className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"
                    aria-hidden
                  />
                  Analyse en cours…
                </>
              ) : (
                "Lancer l’analyse"
              )}
            </button>
          </form>
        </div>

        <aside
          className="min-h-[min(70vh,520px)] lg:sticky lg:top-6 lg:max-h-[calc(100vh-2rem)] lg:overflow-y-auto lg:pl-1"
          aria-label="Résultats de l'analyse"
        >
          <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-text-muted">Résultat</p>

          {loading && progress && progress.total > 0 ? (
            <div className="space-y-4" role="status" aria-busy>
              <div className="rounded-2xl border border-border-default bg-surface p-6 shadow-md shadow-black/20">
                <div className="flex items-center justify-between gap-3">
                  <p className="text-sm font-medium text-text-primary">
                    {progress.processed} / {progress.total} emails traités
                  </p>
                  <span
                    key={progress.processed}
                    className="animate-batch-pop rounded-full bg-ai-glow px-2.5 py-1 text-xs font-semibold text-ai-primary"
                  >
                    +{Math.min(BATCH_CHUNK_SIZE_HINT, progress.processed)} emails analysés
                  </span>
                </div>
                <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-bg-tertiary">
                  <div
                    className="h-full rounded-full bg-ai-primary transition-all duration-500 ease-out"
                    style={{ width: `${Math.min(100, (progress.processed / progress.total) * 100)}%` }}
                  />
                </div>
                <p className="mt-3 text-center text-xs text-text-muted">
                  {LOADING_STATUS_MESSAGES[statusMsgIndex]}
                </p>
              </div>

              {partialBlock ? (
                <AnalysisDashboard
                  projectName={project.trim() || "…"}
                  analysis={partialBlock}
                />
              ) : null}
            </div>
          ) : null}

          {loading && !(progress && progress.total > 0) ? (
            <div
              className="animate-skeleton-panel rounded-2xl border border-border-default bg-surface p-6 shadow-md shadow-black/20"
              role="status"
              aria-busy
            >
              <div className="h-4 w-1/3 rounded-lg bg-bg-tertiary" />
              <div className="mt-4 space-y-3">
                <div className="h-3 w-full rounded bg-border-subtle" />
                <div className="h-3 w-5/6 rounded bg-border-subtle" />
                <div className="h-3 w-4/6 rounded bg-border-subtle" />
              </div>
              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="h-24 rounded-xl bg-border-subtle" />
                <div className="h-24 rounded-xl bg-border-subtle" />
              </div>
              <p className="mt-6 text-center text-sm text-text-muted">Analyse en cours…</p>
            </div>
          ) : null}

          {showPlaceholder && !loading ? (
            <div className="rounded-2xl border border-dashed border-border-default bg-bg-tertiary/30 px-6 py-12 text-center">
              <p className="text-sm text-text-muted">
                Les résultats s&apos;affichent ici après le lancement de l&apos;analyse.
              </p>
            </div>
          ) : null}

          {showResultPanel && !loading ? (
            <div key={resultPanelKey} className="animate-result-panel space-y-6">
              {error ? (
                <div
                  role="alert"
                  className="rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger shadow-sm"
                >
                  <p className="font-medium">Erreur</p>
                  <p className="mt-1 whitespace-pre-wrap">{error}</p>
                </div>
              ) : null}

              {runSuccess && report && !error ? (
                <div
                  className="rounded-2xl border border-success/30 bg-success/10 px-4 py-3 text-sm text-success shadow-sm"
                  role="status"
                >
                  <p className="font-medium">Analyse terminée</p>
                  <p className="mt-0.5 text-success/90">
                    {reportKeys.length} projet{reportKeys.length > 1 ? "s" : ""} dans le rapport.
                  </p>
                </div>
              ) : null}

              {isEmptyReport && !error ? (
                <div
                  className="rounded-2xl border border-border-default bg-surface p-5 shadow-md shadow-black/20"
                  role="status"
                >
                  <p className="text-base font-semibold text-text-primary">Aucun email trouvé</p>
                  <p className="mt-2 text-sm leading-relaxed text-text-secondary">
                    {typeof report?._message === "string" && report._message.trim() ? (
                      report._message
                    ) : (
                      <>
                        La recherche IMAP n&apos;a retourné aucun message dont le sujet ou le corps contient le texte{" "}
                        <span className="rounded-md bg-bg-tertiary px-1.5 py-0.5 font-medium text-text-primary">
                          {project.trim() || "…"}
                        </span>{" "}
                        sur la période choisie.
                      </>
                    )}
                  </p>
                  {typeof report?._imap_folder === "string" || typeof report?._days_back === "number" ? (
                    <p className="mt-2 text-xs text-text-muted">
                      {typeof report._imap_folder === "string" ? (
                        <>Dossier IMAP : {report._imap_folder}. </>
                      ) : null}
                      {typeof report._days_back === "number" ? (
                        <>Fenêtre IMAP : {report._days_back} jour{report._days_back > 1 ? "s" : ""}.</>
                      ) : null}
                    </p>
                  ) : null}
                  <ul className="mt-3 list-inside list-disc text-sm text-text-secondary">
                    <li>Vérifiez l&apos;orthographe du filtre (recherche insensible à la casse).</li>
                    <li>Élargissez la fenêtre : mode &quot;Fenêtre (jours)&quot; avec plus de jours.</li>
                    <li>
                      Confirmez le dossier (<code className="text-xs">IMAP_FOLDER</code> dans{" "}
                      <code className="text-xs">.env</code>, défaut INBOX).
                    </li>
                  </ul>
                </div>
              ) : null}

              {analysisBlock && activeProjectName ? (
                <div className="relative max-w-none">
                  <div className="pb-14">
                    {responseMode === "classic" ? (
                      <AnalysisDashboard
                        projectName={activeProjectName}
                        analysis={analysisBlock}
                        onRequestDraft={requestDraft}
                      />
                    ) : (
                      <ConversationalAssistant
                        sessionKey={`${activeProjectName}-${resultPanelKey}`}
                        projectName={activeProjectName}
                        imapPeriod={periodMode === "preset" && period ? period : null}
                        imapDays={periodMode === "window" ? days : 30}
                        provider={provider}
                        openaiModel={openaiModel}
                        geminiModel={geminiModel}
                        messages={chatMessages}
                        setMessages={setChatMessages}
                        parseResponseJson={parseResponseJson}
                        parseApiError={parseApiError}
                      />
                    )}
                  </div>

                  {responseMode === "classic" ? (
                    <div className="sticky bottom-0 z-20 flex justify-end pb-1 pt-3">
                      <button
                        type="button"
                        onClick={() => setResponseMode("conversational")}
                        className="pointer-events-auto flex h-12 w-12 items-center justify-center rounded-full bg-accent-primary text-white shadow-lg shadow-black/30 transition hover:bg-accent-hover focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary"
                        aria-label="Passer en mode conversationnel"
                        title="Mode conversationnel"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="22"
                          height="22"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          aria-hidden
                        >
                          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                        </svg>
                      </button>
                    </div>
                  ) : (
                    <div className="sticky bottom-0 z-20 flex justify-end pb-1 pt-3">
                      <button
                        type="button"
                        onClick={() => setResponseMode("classic")}
                        className="pointer-events-auto flex h-12 w-12 items-center justify-center rounded-full border border-border-default bg-surface text-text-primary shadow-lg shadow-black/30 transition hover:bg-bg-tertiary focus:outline-none focus:ring-2 focus:ring-accent-primary focus:ring-offset-2 focus:ring-offset-bg-primary"
                        aria-label="Revenir au mode classique"
                        title="Mode classique"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="22"
                          height="22"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          aria-hidden
                        >
                          <rect width="18" height="18" x="3" y="3" rx="2" />
                          <path d="M3 9h18" />
                          <path d="M9 21V9" />
                        </svg>
                      </button>
                    </div>
                  )}
                </div>
              ) : null}
            </div>
          ) : null}
        </aside>
      </div>
    </>
  );

  if (saasEnabled && me) {
    return <AppShell me={me}>{inner}</AppShell>;
  }

  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:py-10">
      <div className="mx-auto max-w-7xl">{inner}</div>
    </div>
  );
}
