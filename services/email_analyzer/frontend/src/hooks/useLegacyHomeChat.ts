import { useRef, useState } from "react";
import { apiFetch } from "../apiClient";
import { parseApiError, parseResponseJson } from "../apiUtils";
import type { DraftResult, ProjectAnalysis } from "../analysisTypes";
import type { HomeMessage } from "../components/chat/messageTypes";
import { applyAnalyzeTickToStore, type AnalyzeJobTick } from "../sync-experience/bridgeAnalyzeJob";
import { useSyncStore } from "../store/syncStore";

export type AssistantProvider = "openai" | "gemini" | "none";

// Polling du job d'analyse : /api/analyze renvoie un job_id (202), on interroge
// /api/analyze/{job_id} jusqu'à done/error. Garde chaque requête HTTP courte
// pour éviter les timeouts de passerelle (Nginx / zrok). Porté à l'identique
// depuis l'ancien HomePage.tsx.
// Budget global de polling aligné sur le job_timeout arq (1800s = 30 min,
// WorkerSettings.job_timeout, analysis_tasks.py) des jobs run_analysis_legacy/
// run_analysis_saas derrière /api/analyze, avec la même marge de sécurité que
// DISCOVER_POLL_TIMEOUT_MS (discoverApi.ts) au-dessus du timeout arq dédié de
// la découverte de domaines : sans ça, le client abandonnait et affichait
// "L'analyse prend trop de temps" jusqu'à 6x plus tôt que le vrai timeout
// backend, sur un job qui tournait souvent encore sainement.
const ANALYZE_POLL_INTERVAL_MS = 2500;
const ANALYZE_POLL_TIMEOUT_MS = 35 * 60 * 1000;

const LOADING_STATUS_MESSAGES = [
  "Je récupère vos emails…",
  "Je les analyse…",
  "Je repère ce qui compte vraiment…",
];

// Doit rester identique à EmailProcessor.NO_FILTER_RESULT_KEY (analyzer.py) :
// clé de résultat utilisée par le backend quand l'analyse est lancée sans
// filtre — sert à savoir si les fils de discussion doivent être proposés
// comme sujets cliquables.
const NO_FILTER_RESULT_KEY = "Tous les emails";

async function pollAnalysis(
  jobId: string,
  onTick: (data: Record<string, unknown>) => void,
): Promise<Record<string, unknown>> {
  const deadline = Date.now() + ANALYZE_POLL_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, ANALYZE_POLL_INTERVAL_MS));
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

function pickProjectBlock(
  data: Record<string, unknown>,
  projectFilter: string,
): { name: string | null; block: ProjectAnalysis | null } {
  const keys = Object.keys(data).filter((k) => !k.startsWith("_"));
  if (keys.length === 0) return { name: null, block: null };
  const key = keys.find((k) => k.toLowerCase() === projectFilter.trim().toLowerCase()) || keys[0];
  const block = data[key];
  if (block && typeof block === "object") {
    return { name: key, block: block as ProjectAnalysis };
  }
  return { name: null, block: null };
}

export type LegacyChatTools = {
  periodMode: "window" | "preset";
  setPeriodMode: (mode: "window" | "preset") => void;
  period: string;
  setPeriod: (period: string) => void;
  days: number;
  setDays: (days: number) => void;
  provider: AssistantProvider;
  setProvider: (provider: AssistantProvider) => void;
  openaiModel: string;
  setOpenaiModel: (model: string) => void;
  geminiModel: string;
  setGeminiModel: (model: string) => void;
};

// Câblage backend de l'accueil chat-first en mode legacy (.env, sans DB) :
// même endpoints qu'avant la refonte (/api/analyze + polling, /api/chat,
// /api/draft), juste réinterprétés en messages de chat plutôt qu'en un
// formulaire + panneau de résultat séparé. Premier message envoyé = filtre
// projet, qui lance une analyse ; une fois qu'une carte d'analyse existe,
// les messages suivants continuent la conversation sur ce projet via
// /api/chat (décision utilisateur : chaque relance/drill-down empile une
// nouvelle carte plutôt que de remplacer le panneau).
export function useLegacyHomeChat() {
  const [messages, setMessages] = useState<HomeMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);

  const [periodMode, setPeriodMode] = useState<"window" | "preset">("window");
  const [period, setPeriod] = useState("7");
  const [days, setDays] = useState(30);
  const [provider, setProvider] = useState<AssistantProvider>("gemini");
  const [openaiModel, setOpenaiModel] = useState("gpt-4o-mini");
  const [geminiModel, setGeminiModel] = useState("gemini-2.5-flash");

  const lastFilterRef = useRef("");
  // Historique texte seul envoyé à /api/chat — remis à zéro à chaque nouvelle
  // analyse puisque le contexte projet change (voir ConversationalAssistant,
  // dont c'est le même contrat de requête).
  const chatThreadRef = useRef<Array<{ role: "user" | "assistant"; content: string }>>([]);

  function lastActiveProjectName(): string | null {
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.kind === "analysis") return m.projectName;
    }
    return null;
  }

  async function launchAnalysis(projectFilter: string, overrideDays?: number) {
    lastFilterRef.current = projectFilter;
    chatThreadRef.current = [];
    if (overrideDays) {
      setPeriodMode("window");
      setDays(overrideDays);
    }

    const statusId = crypto.randomUUID();
    const partialId = `${statusId}-partial`;
    let tickCount = 0;
    setAnalyzing(true);
    setMessages((prev) => [
      ...prev,
      { id: statusId, role: "assistant", kind: "status", content: LOADING_STATUS_MESSAGES[0] },
    ]);

    try {
      const body = {
        project: projectFilter.trim(),
        period: overrideDays ? null : periodMode === "preset" && period ? period : null,
        days: overrideDays ?? (periodMode === "window" ? days : 30),
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
        tickCount += 1;
        const tickProgress = tick.progress as { processed?: number; total?: number } | undefined;
        const progress =
          tickProgress && typeof tickProgress.total === "number" && tickProgress.total > 0
            ? { processed: tickProgress.processed ?? 0, total: tickProgress.total }
            : undefined;

        setMessages((prev) => {
          let next = prev.map((m) =>
            m.id === statusId && m.kind === "status"
              ? { ...m, content: LOADING_STATUS_MESSAGES[tickCount % LOADING_STATUS_MESSAGES.length], progress }
              : m,
          );
          const partial = tick.partial as Record<string, unknown> | null | undefined;
          if (partial && typeof partial === "object") {
            const { block } = pickProjectBlock(partial, projectFilter);
            if (block) {
              next = [
                ...next.filter((m) => m.id !== partialId),
                {
                  id: partialId,
                  role: "assistant",
                  kind: "analysis",
                  projectName: projectFilter || NO_FILTER_RESULT_KEY,
                  analysis: block,
                  allowThreadDrilldown: false,
                },
              ];
            }
          }
          return next;
        });
      });

      setMessages((prev) => prev.filter((m) => m.id !== statusId && m.id !== partialId));

      const { name, block } = pickProjectBlock(data, projectFilter);
      const keys = Object.keys(data).filter((k) => !k.startsWith("_"));
      const isEmpty = keys.length === 0 || data._empty === true || !block || !name;

      if (isEmpty) {
        const effectiveDays = overrideDays ?? (periodMode === "window" ? days : Number(period) || 0);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            kind: "empty-report",
            info: {
              message: typeof data._message === "string" ? data._message : undefined,
              imapFolder: typeof data._imap_folder === "string" ? data._imap_folder : undefined,
              daysBack: typeof data._days_back === "number" ? data._days_back : undefined,
              widenDayOptions: [60, 90, 120, 240].filter((n) => n > effectiveDays),
            },
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            kind: "analysis",
            projectName: name,
            analysis: block,
            allowThreadDrilldown: name === NO_FILTER_RESULT_KEY,
          },
        ]);
      }
    } catch (err) {
      useSyncStore.getState().reset();
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== statusId && m.id !== partialId),
        {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "error",
          content: err instanceof Error ? err.message : "Erreur inconnue",
        },
      ]);
    } finally {
      setAnalyzing(false);
    }
  }

  // Renvoie `false` uniquement quand le backend répond 404 ("aucune donnée
  // disponible pour ce filtre", voir api/main.py::chat) — dans ce cas précis
  // l'appelant (`send`) se rabat sur une vraie analyse (`launchAnalysis`),
  // seul recours quand rien n'est encore disponible. Toute autre erreur est
  // affichée directement comme message d'erreur, pas de repli.
  async function sendChatReply(text: string, projectName: string): Promise<boolean> {
    const nextThread = [...chatThreadRef.current, { role: "user" as const, content: text }];
    chatThreadRef.current = nextThread;
    setBusy(true);
    try {
      const body = {
        project_name: projectName,
        period: periodMode === "preset" && period ? period : null,
        days: periodMode === "window" ? days : 30,
        messages: nextThread,
        assistant_provider: provider,
        openai_model: openaiModel,
        gemini_model: geminiModel,
      };
      const res = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await parseResponseJson(res);
      if (!res.ok) {
        if (res.status === 404) {
          chatThreadRef.current = chatThreadRef.current.slice(0, -1);
          return false;
        }
        throw new Error(parseApiError(data));
      }
      const msg = (data as { message?: string }).message;
      if (!msg) throw new Error("Réponse vide");
      chatThreadRef.current = [...nextThread, { role: "assistant", content: msg }];
      setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "assistant", kind: "text", content: msg }]);
      return true;
    } catch (e) {
      chatThreadRef.current = chatThreadRef.current.slice(0, -1);
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "assistant", kind: "error", content: e instanceof Error ? e.message : "Erreur" },
      ]);
      return true;
    } finally {
      setBusy(false);
    }
  }

  // Écrire dans le chat ne doit pas relancer une analyse complète
  // (`/api/analyze`, job de fond) si une réponse peut déjà être construite à
  // partir des données disponibles : avec un projet actif, `/api/chat` sert
  // désormais le cache local en priorité (voir
  // analyzer.py::fetch_last_n_emails_for_chat) ; sans projet actif, le texte
  // tapé est tenté comme filtre projet sur ce même chemin cache-first avant
  // de se rabattre sur une vraie analyse — seulement si rien n'est
  // disponible pour ce filtre (404).
  function send(text: string) {
    const active = lastActiveProjectName();
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", kind: "text", content: text }]);

    if (active) {
      void sendChatReply(text, active);
      return;
    }

    const filter = text.trim();
    if (!filter) {
      void launchAnalysis(text);
      return;
    }

    void (async () => {
      const answered = await sendChatReply(text, filter);
      if (!answered) {
        void launchAnalysis(text);
      }
    })();
  }

  function widenDays(days: number) {
    void launchAnalysis(lastFilterRef.current, days);
  }

  function selectThread(subject: string) {
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), role: "user", kind: "text", content: subject }]);
    void launchAnalysis(subject);
  }

  async function requestDraft(projectName: string, analysis: ProjectAnalysis): Promise<DraftResult> {
    const res = await apiFetch("/api/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_name: projectName, analysis }),
    });
    const data = await parseResponseJson(res);
    if (!res.ok) throw new Error(parseApiError(data));
    return data as DraftResult;
  }

  const tools: LegacyChatTools = {
    periodMode,
    setPeriodMode,
    period,
    setPeriod,
    days,
    setDays,
    provider,
    setProvider,
    openaiModel,
    setOpenaiModel,
    geminiModel,
    setGeminiModel,
  };

  return {
    messages,
    busy: busy || analyzing,
    loaded: true,
    send,
    widenDays,
    selectThread,
    requestDraft,
    tools,
  };
}
