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

type AssistantProvider = "openai" | "gemini" | "none";

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
  const [responseMode, setResponseMode] = useState<"classic" | "conversational">("classic");
  const [chatMessages, setChatMessages] = useState<ChatTurn[]>([]);

  const { me } = useSaasSession(saasEnabled, sessionTick);

  useEffect(() => {
    setChatMessages([]);
    setResponseMode("classic");
  }, [activeProjectName, resultPanelKey]);

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
    setLoading(true);
    try {
      const body = {
        project: project.trim(),
        period: periodMode === "preset" && period ? period : null,
        days: periodMode === "window" ? days : 30,
        assistant_provider: provider,
        openai_model: openaiModel,
        gemini_model: geminiModel,
      };
      const res = await apiFetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = (await parseResponseJson(res)) as Record<string, unknown>;
      if (!res.ok) {
        throw new Error(parseApiError(data));
      }
      setReport(data);
      const { name, block } = pickProjectBlock(data, project);
      setActiveProjectName(name);
      setAnalysisBlock(block);
      const hasKeys = Object.keys(data).filter((k) => !k.startsWith("_")).length > 0;
      setRunSuccess(hasKeys && block !== null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur inconnue");
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
      <div className="flex min-h-screen flex-col items-center justify-center bg-stone-100/80 px-4">
        <p className="text-sm text-slate-500">Chargement du compte…</p>
      </div>
    );
  }

  if (saasEnabled && !getAccessToken()) {
    return <Navigate to="/login" replace />;
  }

  const inner = (
    <>
      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Analyse d&apos;emails</h1>
        <p className="mt-1 text-sm text-slate-500">
          {saasEnabled ? (
            <>
              Mode SaaS — boîte IMAP et quotas par organisation. Configurez la boîte dans{" "}
              <Link to="/settings" className="font-medium text-slate-700 underline decoration-slate-300 underline-offset-2 hover:text-slate-900">
                Paramètres
              </Link>
              .
            </>
          ) : (
            <>
              Lancez une analyse depuis votre boîte (identifiants IMAP dans le fichier{" "}
              <code className="rounded bg-stone-200/60 px-1 text-xs">.env</code> du dépôt).
            </>
          )}
        </p>
      </header>

      {billingReturnBanner ? (
        <div
          className="mb-6 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-center text-sm text-emerald-900"
          role="status"
        >
          Retour depuis le paiement : si le montant est validé, votre organisation sera activée sous peu
          (notification serveur CinetPay).
        </div>
      ) : null}

      {healthError ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900"
        >
          Impossible de joindre l&apos;API ({healthError}). Vérifiez que{" "}
          <code className="text-xs">uvicorn</code> tourne sur le port 8000.
        </div>
      ) : null}

      {!saasEnabled && health && !health.imap_configured ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-950"
        >
          IMAP non configuré : renseignez <code className="font-mono text-xs">IMAP_USER</code> et{" "}
          <code className="font-mono text-xs">IMAP_PASSWORD</code> dans{" "}
          <code className="font-mono text-xs">.env</code> à la racine du projet.
        </div>
      ) : null}

      {saasEnabled && me ? (
        <div
          role="status"
          className="mb-6 rounded-2xl border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-950"
        >
          Organisation <strong>{me.tenants.find((t) => t.id === me.active_tenant_id)?.name ?? ""}</strong> — statut :{" "}
          <strong>{me.tenants.find((t) => t.id === me.active_tenant_id)?.status ?? ""}</strong>. Configurez la boîte
          IMAP dans{" "}
          <Link to="/settings" className="font-medium text-sky-950 underline underline-offset-2">
            Paramètres
          </Link>{" "}
          si ce n&apos;est pas encore fait.
        </div>
      ) : null}

      <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] lg:gap-10 xl:grid-cols-[380px_minmax(0,1fr)]">
        <div className="lg:max-w-md">
          <form
            onSubmit={runAnalyze}
            className="rounded-2xl border border-stone-200/80 bg-white p-6 shadow-sm shadow-stone-200/40"
          >
            <label className="block">
              <span className="mb-1 block text-sm font-medium text-slate-700">Projet (filtre)</span>
              <input
                type="text"
                value={project}
                onChange={(e) => setProject(e.target.value)}
                required
                placeholder="Ex. BBCI, Nom du client…"
                className="mt-1 w-full rounded-xl border border-stone-200 bg-stone-50/50 px-3 py-2.5 text-sm outline-none ring-slate-300 transition placeholder:text-slate-400 focus:border-slate-300 focus:bg-white focus:ring-2"
              />
            </label>

            <div className="mt-5">
              <span className="text-sm font-medium text-slate-700">Période</span>
              <div className="mt-2 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => setPeriodMode("window")}
                  className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
                    periodMode === "window"
                      ? "bg-slate-800 text-white"
                      : "bg-stone-100 text-slate-600 hover:bg-stone-200"
                  }`}
                >
                  Fenêtre (jours)
                </button>
                <button
                  type="button"
                  onClick={() => setPeriodMode("preset")}
                  className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
                    periodMode === "preset"
                      ? "bg-slate-800 text-white"
                      : "bg-stone-100 text-slate-600 hover:bg-stone-200"
                  }`}
                >
                  Préréglage
                </button>
              </div>
              {periodMode === "window" ? (
                <label className="mt-3 block">
                  <span className="text-xs text-slate-500">Nombre de jours (IMAP)</span>
                  <input
                    type="number"
                    min={1}
                    max={365}
                    value={days}
                    onChange={(e) => setDays(Number(e.target.value))}
                    className="mt-1 w-full rounded-xl border border-stone-200 bg-stone-50/50 px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-slate-300"
                  />
                </label>
              ) : (
                <select
                  value={period}
                  onChange={(e) => setPeriod(e.target.value)}
                  className="mt-3 w-full rounded-xl border border-stone-200 bg-stone-50/50 px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-slate-300"
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
              <span className="text-sm font-medium text-slate-700">Assistant LLM</span>
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
                      provider === p ? "bg-slate-800 text-white" : "bg-stone-100 text-slate-600 hover:bg-stone-200"
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
              className="mt-4 text-sm text-slate-500 underline decoration-slate-300 underline-offset-2 hover:text-slate-700"
            >
              {advancedOpen ? "Masquer" : "Avancé"} — modèles
            </button>
            {advancedOpen ? (
              <div className="mt-3 space-y-3 rounded-xl border border-stone-100 bg-stone-50/50 p-3">
                <label className="block text-xs">
                  <span className="text-slate-600">Modèle OpenAI</span>
                  <input
                    type="text"
                    value={openaiModel}
                    onChange={(e) => setOpenaiModel(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-stone-200 px-2 py-1.5 text-sm"
                  />
                </label>
                <label className="block text-xs">
                  <span className="text-slate-600">Modèle Gemini</span>
                  <input
                    type="text"
                    value={geminiModel}
                    onChange={(e) => setGeminiModel(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-stone-200 px-2 py-1.5 text-sm"
                  />
                </label>
              </div>
            ) : null}

            <button
              type="submit"
              disabled={loading || !project.trim()}
              className="mt-6 flex w-full items-center justify-center gap-2 rounded-xl bg-slate-800 py-3 text-sm font-semibold text-white shadow-md shadow-slate-300/40 transition hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
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
          <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Résultat</p>

          {loading ? (
            <div
              className="animate-skeleton-panel rounded-2xl border border-stone-200/80 bg-white p-6 shadow-md shadow-stone-200/30"
              role="status"
              aria-busy
            >
              <div className="h-4 w-1/3 rounded-lg bg-stone-200" />
              <div className="mt-4 space-y-3">
                <div className="h-3 w-full rounded bg-stone-100" />
                <div className="h-3 w-5/6 rounded bg-stone-100" />
                <div className="h-3 w-4/6 rounded bg-stone-100" />
              </div>
              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <div className="h-24 rounded-xl bg-stone-100" />
                <div className="h-24 rounded-xl bg-stone-100" />
              </div>
              <p className="mt-6 text-center text-sm text-slate-500">Analyse en cours…</p>
            </div>
          ) : null}

          {showPlaceholder && !loading ? (
            <div className="rounded-2xl border border-dashed border-stone-300 bg-stone-50/80 px-6 py-12 text-center">
              <p className="text-sm text-slate-500">
                Les résultats s&apos;affichent ici après le lancement de l&apos;analyse.
              </p>
            </div>
          ) : null}

          {showResultPanel && !loading ? (
            <div key={resultPanelKey} className="animate-result-panel space-y-6">
              {error ? (
                <div
                  role="alert"
                  className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900 shadow-sm"
                >
                  <p className="font-medium">Erreur</p>
                  <p className="mt-1 whitespace-pre-wrap">{error}</p>
                </div>
              ) : null}

              {runSuccess && report && !error ? (
                <div
                  className="rounded-2xl border border-emerald-200 bg-emerald-50/90 px-4 py-3 text-sm text-emerald-950 shadow-sm"
                  role="status"
                >
                  <p className="font-medium">Analyse terminée</p>
                  <p className="mt-0.5 text-emerald-900/90">
                    {reportKeys.length} projet{reportKeys.length > 1 ? "s" : ""} dans le rapport.
                  </p>
                </div>
              ) : null}

              {isEmptyReport && !error ? (
                <div
                  className="rounded-2xl border border-stone-200 bg-white p-5 shadow-md shadow-stone-200/40"
                  role="status"
                >
                  <p className="text-base font-semibold text-slate-900">Aucun email trouvé</p>
                  <p className="mt-2 text-sm leading-relaxed text-slate-600">
                    {typeof report?._message === "string" && report._message.trim() ? (
                      report._message
                    ) : (
                      <>
                        La recherche IMAP n&apos;a retourné aucun message dont le sujet ou le corps contient le texte{" "}
                        <span className="rounded-md bg-stone-100 px-1.5 py-0.5 font-medium text-slate-800">
                          {project.trim() || "…"}
                        </span>{" "}
                        sur la période choisie.
                      </>
                    )}
                  </p>
                  {typeof report?._imap_folder === "string" || typeof report?._days_back === "number" ? (
                    <p className="mt-2 text-xs text-slate-500">
                      {typeof report._imap_folder === "string" ? (
                        <>Dossier IMAP : {report._imap_folder}. </>
                      ) : null}
                      {typeof report._days_back === "number" ? (
                        <>Fenêtre IMAP : {report._days_back} jour{report._days_back > 1 ? "s" : ""}.</>
                      ) : null}
                    </p>
                  ) : null}
                  <ul className="mt-3 list-inside list-disc text-sm text-slate-600">
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
                        className="pointer-events-auto flex h-12 w-12 items-center justify-center rounded-full bg-slate-800 text-white shadow-lg shadow-slate-400/40 transition hover:bg-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2"
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
                        className="pointer-events-auto flex h-12 w-12 items-center justify-center rounded-full border border-stone-200 bg-white text-slate-800 shadow-lg shadow-stone-300/50 transition hover:bg-stone-50 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2"
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
