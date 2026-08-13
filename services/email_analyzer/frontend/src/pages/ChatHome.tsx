import React, { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import AppShell from "../components/AppShell";
import GreetingCharacter from "../components/GreetingCharacter";
import ChatThread from "../components/chat/ChatThread";
import type { ChatSuggestion } from "../components/chat/ChatThread";
import type { Health } from "../apiUtils";
import type { Me } from "../SaasPanels";
import DiscoverProjectsModal from "../DiscoverProjectsModal";
import DiscoverTicketsModal from "../DiscoverTicketsModal";
import { syncAllKnowledge } from "../knowledgeApi";
import ConversationHistoryPanel from "../components/chat/ConversationHistoryPanel";
import { useLegacyHomeChat } from "../hooks/useLegacyHomeChat";
import { useLegacyRecentEmails } from "../hooks/useLegacyRecentEmails";
import { useSaasHomeChat } from "../hooks/useSaasHomeChat";
import { formatSince, greetingName, useBrief } from "../hooks/useBrief";
import type { ActionOut } from "../actionsApi";
import { fetchTimeline, type TimelineEvent } from "../timelineApi";
import {
  getTimeOfDaySalutation,
  pickVariant,
  SAAS_BRIEF_SUBTITLES,
  SAAS_LOADING_SUBTITLES,
  LEGACY_SUBTITLES,
} from "../greeting";

type QuickAction = {
  icon: string;
  label: string;
  description: string;
  badge?: number;
  onClick?: () => void;
};

function formatDeadline(deadline: string): string {
  try {
    return new Intl.DateTimeFormat("fr-FR", { day: "2-digit", month: "long" }).format(
      new Date(deadline),
    );
  } catch {
    return deadline;
  }
}

// Nombre max d'actions recommandées affichées sur l'accueil — au-delà, à
// consulter dans la page Actions dédiée (cohérent avec les cartes rapides
// existantes, aussi limitées à quelques entrées).
const MAX_RECOMMENDED_ACTIONS_SHOWN = 3;

function LegacyToolsPanel({
  tools,
}: {
  tools: ReturnType<typeof useLegacyHomeChat>["tools"];
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <div className="rounded-lg border border-border-subtle bg-bg-tertiary/40 p-4">
      <div>
        <span className="text-sm font-medium text-text-secondary">Période</span>
        <div className="mt-2 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => tools.setPeriodMode("window")}
            className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
              tools.periodMode === "window"
                ? "bg-accent-primary text-white"
                : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
            }`}>
            Fenêtre (jours)
          </button>
          <button
            type="button"
            onClick={() => tools.setPeriodMode("preset")}
            className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
              tools.periodMode === "preset"
                ? "bg-accent-primary text-white"
                : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
            }`}>
            Préréglage
          </button>
        </div>
        {tools.periodMode === "window" ? (
          <label className="mt-3 block">
            <span className="text-xs text-text-muted">
              Nombre de jours (IMAP)
            </span>
            <input
              type="number"
              min={1}
              max={365}
              value={tools.days}
              onChange={(e) => tools.setDays(Number(e.target.value))}
              className="mt-1 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2 text-sm text-text-primary outline-none focus:ring-2 focus:ring-accent-primary"
            />
          </label>
        ) : (
          <select
            value={tools.period}
            onChange={(e) => tools.setPeriod(e.target.value)}
            className="mt-3 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2.5 text-sm text-text-primary outline-none focus:ring-2 focus:ring-accent-primary">
            <option value="today">Aujourd&apos;hui</option>
            <option value="yesterday">Hier</option>
            <option value="3">3 derniers jours</option>
            <option value="7">7 derniers jours</option>
            <option value="11">11 derniers jours</option>
            <option value="90">90 derniers jours</option>
            <option value="120">120 derniers jours</option>
            <option value="240">240 derniers jours</option>
          </select>
        )}
      </div>

      <div className="mt-4">
        <span className="text-sm font-medium text-text-secondary">
          Assistant LLM
        </span>
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
              onClick={() => tools.setProvider(p)}
              className={`rounded-full px-3 py-1.5 text-sm font-medium transition ${
                tools.provider === p
                  ? "bg-accent-primary text-white"
                  : "bg-bg-tertiary text-text-secondary hover:bg-border-default"
              }`}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <button
        type="button"
        onClick={() => setAdvancedOpen(!advancedOpen)}
        className="mt-4 text-sm text-text-muted underline decoration-border-default underline-offset-2 hover:text-text-secondary">
        {advancedOpen ? "Masquer" : "Avancé"} — modèles
      </button>
      {advancedOpen ? (
        <div className="mt-3 space-y-3 rounded-xl border border-border-subtle bg-bg-tertiary/40 p-3">
          <label className="block text-xs">
            <span className="text-text-secondary">Modèle OpenAI</span>
            <input
              type="text"
              value={tools.openaiModel}
              onChange={(e) => tools.setOpenaiModel(e.target.value)}
              className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-1.5 text-sm text-text-primary"
            />
          </label>
          <label className="block text-xs">
            <span className="text-text-secondary">Modèle Gemini</span>
            <input
              type="text"
              value={tools.geminiModel}
              onChange={(e) => tools.setGeminiModel(e.target.value)}
              className="mt-1 w-full rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-1.5 text-sm text-text-primary"
            />
          </label>
        </div>
      ) : null}
    </div>
  );
}

// Accueil chat-first, partagé entre le mode legacy (.env, sans DB) et le
// mode SaaS. Legacy : câblé sur useLegacyHomeChat (/api/analyze + polling,
// /api/chat, /api/draft). SaaS : câblé sur useSaasHomeChat
// (/api/assistant/chat, persistant) + useBrief (/api/brief, alimente la
// salutation et les cartes d'action) — reprend le contenu fonctionnel des
// anciens BriefPage.tsx/PortfolioAssistant.tsx/AssistantPage.tsx, retirés
// (voir progress-tracker.md, Unit 41).
export default function ChatHome({
  mode,
  me,
  saasEnabled,
  health,
  healthError,
  billingReturnBanner,
}: {
  mode: "legacy" | "saas";
  me: Me | null;
  saasEnabled: boolean;
  health: Health | null;
  healthError: string | null;
  billingReturnBanner: boolean;
}) {
  const [toolsOpen, setToolsOpen] = useState(false);
  const [showDiscoverModal, setShowDiscoverModal] = useState(false);
  const [showDiscoverTicketsModal, setShowDiscoverTicketsModal] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [discoverSuccess, setDiscoverSuccess] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [syncError, setSyncError] = useState<string | null>(null);
  const [syncMessage, setSyncMessage] = useState<string | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const initialQuestion = searchParams.get("q") ?? undefined;
  const [recentEmails, setRecentEmails] = useState<TimelineEvent[]>([]);

  const legacyChat = useLegacyHomeChat();
  const saasChat = useSaasHomeChat(mode === "saas", initialQuestion);
  const brief = useBrief(mode === "saas");
  const chat = mode === "legacy" ? legacyChat : saasChat;
  const legacyRecentEmails = useLegacyRecentEmails(mode === "legacy");

  useEffect(() => {
    if (mode !== "saas") return;
    let cancelled = false;
    fetchTimeline(20)
      .then((events) => {
        if (cancelled) return;
        setRecentEmails(events.filter((e) => e.type === "email").slice(0, 4));
      })
      .catch(() => {
        // Suggestions best-effort : une timeline indisponible ne doit pas
        // bloquer le chat.
      });
    return () => {
      cancelled = true;
    };
  }, [mode]);

  function handleSend(text: string) {
    chat.send(text);
    if (initialQuestion) {
      searchParams.delete("q");
      setSearchParams(searchParams, { replace: true });
    }
  }

  async function handleSyncKnowledge() {
    setSyncing(true);
    setSyncError(null);
    setSyncMessage(null);
    try {
      const result = await syncAllKnowledge();
      setSyncMessage(
        `${result.synced_projects}/${result.total_projects} projet(s) synchronisé(s).`,
      );
    } catch (err) {
      setSyncError(err instanceof Error ? err.message : "Échec de la synchronisation.");
    } finally {
      setSyncing(false);
    }
  }

  function handleProjectsProcessed(created: number, updated: number) {
    setShowDiscoverModal(false);
    const parts: string[] = [];
    if (created > 0)
      parts.push(
        `${created} projet${created > 1 ? "s" : ""} ajouté${created > 1 ? "s" : ""}`,
      );
    if (updated > 0)
      parts.push(`${updated} projet${updated > 1 ? "s" : ""} mis à jour`);
    setDiscoverSuccess(parts.length > 0 ? `${parts.join(", ")}.` : null);
    void brief.reload();
  }

  function handleTicketsProcessed(created: number, updated: number, attached: number) {
    setShowDiscoverTicketsModal(false);
    const parts: string[] = [];
    if (created > 0)
      parts.push(
        `${created} projet${created > 1 ? "s" : ""} créé${created > 1 ? "s" : ""}`,
      );
    if (attached > 0)
      parts.push(`${attached} ticket${attached > 1 ? "s" : ""} rattaché${attached > 1 ? "s" : ""}`);
    if (updated > 0)
      parts.push(`${updated} projet${updated > 1 ? "s" : ""} mis à jour`);
    setDiscoverSuccess(parts.length > 0 ? `${parts.join(", ")}.` : null);
    void brief.reload();
  }

  const recommendedActions: ActionOut[] = (brief.brief?.recommended_actions ?? []).slice(
    0,
    MAX_RECOMMENDED_ACTIONS_SHOWN,
  );

  const quickActions: QuickAction[] =
    mode === "saas"
      ? [
          {
            icon: "📬",
            label: "Analyser mes emails",
            description: "Découvrir de nouveaux projets à partir de la boîte.",
            onClick: () => {
              setDiscoverSuccess(null);
              setShowDiscoverModal(true);
            },
          },
          {
            icon: "🎫",
            label: "Détecter les tickets",
            description: "Repérer les réclamations client à partir des sujets.",
            onClick: () => {
              setDiscoverSuccess(null);
              setShowDiscoverTicketsModal(true);
            },
          },
          {
            icon: "✅",
            label: "Actions en attente",
            description: "Voir ce qui reste à traiter cette semaine.",
            badge: brief.brief?.counts.pending_actions,
          },
          {
            icon: "⚠️",
            label: "Projets à risque",
            description: "Repérer ce qui demande votre attention.",
            badge: brief.brief?.counts.at_risk_projects,
          },
        ]
      : [];

  const chatSuggestions: ChatSuggestion[] =
    mode === "saas"
      ? recentEmails.map((e) => ({
          key: `${e.project_id}-${e.at}`,
          label: `📧 ${e.label}`,
          onClick: () =>
            handleSend(`Que dois-je savoir sur l'email "${e.label}" (projet ${e.project_name}) ?`),
        }))
      : legacyRecentEmails.map((e, i) => ({
          key: `${e.date}-${i}`,
          label: `📧 ${e.subject}`,
          onClick: () => handleSend(`Que dois-je savoir sur l'email "${e.subject}" ?`),
        }));

  const salutation = getTimeOfDaySalutation();
  const greeting =
    mode === "saas" && me
      ? {
          title: `${salutation} ${greetingName(me.email)}`,
          subtitle: brief.brief
            ? pickVariant(SAAS_BRIEF_SUBTITLES)(formatSince(brief.brief.since))
            : pickVariant(SAAS_LOADING_SUBTITLES),
        }
      : {
          title: salutation,
          subtitle: pickVariant(LEGACY_SUBTITLES),
        };

  // Une fois la conversation démarrée, l'avatar + les cartes rejoignent une
  // colonne gauche persistante au lieu de disparaître (voir rendu plus bas) —
  // uniquement en mode saas, seul mode où quickActions n'est pas vide.
  const inConversation = chat.loaded && chat.messages.length > 0;
  const showSidebar = inConversation && mode === "saas";

  function renderQuickActions(containerClassName: string) {
    return (
      <div className={containerClassName}>
        {quickActions.map((action) => (
          <button
            key={action.label}
            type="button"
            onClick={action.onClick}
            disabled={!action.onClick}
            className="relative rounded-lg border border-border-default bg-surface p-4 text-left shadow-sm transition hover:bg-bg-tertiary disabled:cursor-default disabled:hover:bg-surface">
            {action.badge ? (
              <span className="absolute right-3 top-3 rounded-full bg-accent-soft px-2 py-0.5 text-xs font-semibold text-accent-primary">
                {action.badge}
              </span>
            ) : null}
            <p className="text-lg" aria-hidden>
              {action.icon}
            </p>
            <p className="mt-1 text-sm font-semibold text-text-primary">
              {action.label}
            </p>
            <p className="mt-1 text-xs text-text-muted">
              {action.description}
            </p>
          </button>
        ))}
      </div>
    );
  }

  const toolbar =
    mode === "saas" ? (
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={() => void saasChat.newConversation()}
          className="rounded-xl border border-border-default bg-surface px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary">
          + Nouvelle conversation
        </button>
        <button
          type="button"
          onClick={() => setShowHistory(true)}
          className="rounded-xl border border-border-default bg-surface px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary">
          Historique
        </button>
        <button
          type="button"
          onClick={() => void handleSyncKnowledge()}
          disabled={syncing}
          className="rounded-xl border border-border-default bg-surface px-3 py-1.5 text-sm font-medium text-text-secondary hover:bg-bg-tertiary disabled:cursor-default disabled:opacity-60">
          {syncing ? "Synchronisation…" : "Synchroniser"}
        </button>
      </div>
    ) : null;

  const chatThreadEl = (
    <ChatThread
      messages={chat.messages}
      busy={chat.busy}
      onSend={handleSend}
      placeholder={
        mode === "legacy"
          ? "Nom de projet (optionnel) ou message… (Entrée pour envoyer)"
          : "Posez une question sur vos projets… (Entrée pour envoyer)"
      }
      onWidenDays={mode === "legacy" ? legacyChat.widenDays : undefined}
      onSelectThread={mode === "legacy" ? legacyChat.selectThread : undefined}
      onRequestDraft={mode === "legacy" ? legacyChat.requestDraft : undefined}
      suggestions={chatSuggestions}
    />
  );

  return (
    <AppShell me={me} saasEnabled={saasEnabled}>
      <div className={`mx-auto flex flex-col gap-6 ${inConversation ? "max-w-5xl" : "max-w-3xl"}`}>
        {billingReturnBanner ? (
          <div
            className="rounded-lg border border-success/30 bg-success/10 px-4 py-3 text-center text-sm text-success"
            role="status">
            Retour depuis le paiement : si le montant est validé, votre
            organisation sera activée sous peu (notification serveur CinetPay).
          </div>
        ) : null}

        {healthError ? (
          <div
            role="alert"
            className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
            Impossible de joindre l&apos;API ({healthError}). Vérifiez que{" "}
            <code className="text-xs">uvicorn</code> tourne sur le port 8000.
          </div>
        ) : null}

        {mode === "legacy" && health && !health.imap_configured ? (
          <div
            role="alert"
            className="rounded-lg border border-warning/30 bg-warning/10 px-4 py-3 text-sm text-warning">
            IMAP non configuré : renseignez{" "}
            <code className="font-mono text-xs">IMAP_USER</code> et{" "}
            <code className="font-mono text-xs">IMAP_PASSWORD</code> dans{" "}
            <code className="font-mono text-xs">.env</code> à la racine du
            projet.
          </div>
        ) : null}

        {discoverSuccess ? (
          <div
            role="status"
            className="rounded-lg border border-success/30 bg-success/10 px-4 py-3 text-sm text-success">
            {discoverSuccess}{" "}
            <Link to="/projects" className="font-medium underline">
              Voir mes projets
            </Link>
          </div>
        ) : null}

        {syncMessage ? (
          <div
            role="status"
            className="rounded-lg border border-success/30 bg-success/10 px-4 py-3 text-sm text-success">
            {syncMessage}
          </div>
        ) : null}

        {syncError ? (
          <div
            role="alert"
            className="rounded-lg border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
            {syncError}
          </div>
        ) : null}

        {!chat.loaded ? (
          <div className="flex flex-col items-center gap-4 pt-4 text-center">
            <GreetingCharacter size={128} active={false} />
            <div>
              <p className="text-lg text-text-secondary">{salutation}</p>
              <h1 className="text-2xl font-semibold tracking-tight text-text-muted">
                Chargement de votre conversation…
              </h1>
            </div>
          </div>
        ) : null}

        {chat.loaded && chat.messages.length === 0 ? (
          <div className="flex flex-col items-center gap-4 pt-4 text-center">
            <GreetingCharacter size={128} active />
            <div>
              <p className="text-lg text-text-secondary">{greeting.title}</p>
              <h1 className="text-2xl font-semibold tracking-tight text-text-primary">
                {greeting.subtitle}
              </h1>
            </div>
          </div>
        ) : null}

        {mode === "saas" && chat.loaded && chat.messages.length === 0 && recommendedActions.length > 0 ? (
          <div className="mx-auto w-full max-w-2xl">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">
              Ce qui mérite votre attention
            </p>
            <div className="flex flex-col gap-2">
              {recommendedActions.map((action) => (
                <button
                  key={action.id}
                  type="button"
                  onClick={() =>
                    handleSend(
                      `Où en est l'action "${action.description}" pour le projet ${action.project_name} ?`,
                    )
                  }
                  className="rounded-lg border border-border-default bg-surface p-3 text-left shadow-sm transition hover:bg-bg-tertiary">
                  <p className="text-sm font-medium text-text-primary">{action.description}</p>
                  <p className="mt-0.5 text-xs text-text-muted">
                    {action.project_name}
                    {action.deadline ? ` · échéance ${formatDeadline(action.deadline)}` : ""}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {mode === "legacy" ? (
          <div>
            <button
              type="button"
              onClick={() => setToolsOpen((v) => !v)}
              className="mb-2 text-sm text-text-muted underline decoration-border-default underline-offset-2 hover:text-text-secondary">
              {toolsOpen ? "Masquer les outils" : "Outils (période, modèle IA)"}
            </button>
            {toolsOpen ? <LegacyToolsPanel tools={legacyChat.tools} /> : null}
          </div>
        ) : null}

        {showSidebar ? (
          <div className="lg:grid lg:grid-cols-[280px_1fr] lg:gap-6">
            <div className="mb-6 flex flex-col items-center gap-4 lg:mb-0 lg:items-stretch">
              <GreetingCharacter size={64} active={false} />
              {renderQuickActions("flex flex-col gap-3")}
            </div>

            <div className="flex flex-col gap-6">
              {toolbar}
              {chatThreadEl}
            </div>
          </div>
        ) : (
          <>
            {toolbar}

            {chatThreadEl}

            {chat.loaded && chat.messages.length === 0
              ? renderQuickActions("grid gap-3 sm:grid-cols-3")
              : null}
          </>
        )}
      </div>

      {showDiscoverModal ? (
        <DiscoverProjectsModal
          onClose={() => setShowDiscoverModal(false)}
          onProjectsProcessed={handleProjectsProcessed}
        />
      ) : null}

      {showDiscoverTicketsModal ? (
        <DiscoverTicketsModal
          onClose={() => setShowDiscoverTicketsModal(false)}
          onTicketsProcessed={handleTicketsProcessed}
        />
      ) : null}

      {showHistory ? (
        <ConversationHistoryPanel
          conversations={saasChat.conversations}
          activeId={saasChat.activeConversationId}
          onSelect={(id) => {
            void saasChat.switchConversation(id);
            setShowHistory(false);
          }}
          onClose={() => setShowHistory(false)}
        />
      ) : null}
    </AppShell>
  );
}
