import { useQueryClient } from "@tanstack/react-query";
import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { ApiError, ApiRequest, SOURCE_LABEL, SourceCode, SourceInfo, Task, api, onConsentRequired } from "./api";
import { ComposeInitial, ComposeModal } from "./components/ComposeModal";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { CAUTION_MAX } from "./schemas";
import { RiskConsentDialog } from "./components/RiskConsentDialog";
import { useTaskUi } from "./components/TaskDrawer";
import { markTokenExpired, useAction, useApplication, useClearBlock, useSources, useStatus, useTokenExpired } from "./hooks";
import Applications from "./pages/Applications";
import Dashboard from "./pages/Dashboard";
import Jobs from "./pages/Jobs";
import SettingsPage from "./pages/SettingsPage";
import Skills from "./pages/Skills";
import { Button, ErrorBox, Notice, formatDate } from "./ui";

// Marché (graphiques, Markdown) est chargé à la demande.
const Market = lazy(() => import("./pages/Market"));

const NAV = [
  { to: "/", label: "Tableau de bord", end: true },
  { to: "/offres", label: "Offres" },
  { to: "/candidatures", label: "Candidatures" },
  { to: "/marche", label: "Marché" },
  { to: "/competences", label: "Compétences" },
  { to: "/parametres", label: "Paramètres" },
];

/** Section de premier niveau de l'URL (`/candidatures/job:1` → `/candidatures`). */
const sectionOf = (pathname: string) => `/${pathname.split("/")[1] ?? ""}`;

/**
 * Titre du document par page (WCAG 2.4.2) et, à chaque changement de section, focus sur le <h1> de la page
 * (`tabIndex=-1`) pour les lecteurs d'écran. Le changement d'élément dans une même section (candidature
 * sélectionnée) ne déplace pas le focus.
 */
function useRouteAnnouncements() {
  const { pathname } = useLocation();
  const section = sectionOf(pathname);
  const previous = useRef(section);
  useEffect(() => {
    const label = NAV.find((item) => item.to === section)?.label;
    document.title = label ? `${label} · LinkedIn Market` : "LinkedIn Market";
    if (previous.current === section) return;
    previous.current = section;
    const main = document.querySelector("main");
    const focusHeading = () => {
      const heading = main?.querySelector<HTMLElement>("h1");
      heading?.focus({ preventScroll: true });
      return !!heading;
    };
    if (!main || focusHeading()) return;
    // Page chargée à la demande (Marché) : attendre l'apparition du titre, 3 s au plus.
    const observer = new MutationObserver(() => { if (focusHeading()) observer.disconnect(); });
    observer.observe(main, { childList: true, subtree: true });
    const timer = setTimeout(() => observer.disconnect(), 3000);
    return () => { observer.disconnect(); clearTimeout(timer); };
  }, [section]);
}

/**
 * Erreurs traitées globalement (frontend/03 § 3.2) : un 401 signifie que le jeton local a changé (redémarrage de
 * l'application) ; requêtes et interrogation périodique sont alors suspendues (`markTokenExpired`) et un seul
 * bandeau s'affiche. Les autres codes restent affichés par le composant appelant, `detail` tel quel.
 */
function useWatchUnauthorized() {
  const client = useQueryClient();
  useEffect(() => {
    const check = (error: unknown) => {
      if (error instanceof ApiError && error.status === 401) {
        markTokenExpired();
        void client.cancelQueries();
      }
    };
    const unsubscribeQueries = client.getQueryCache().subscribe((event) => {
      if (event.type === "updated" && event.action.type === "error") check(event.action.error);
    });
    const unsubscribeMutations = client.getMutationCache().subscribe((event) => {
      if (event.type === "updated" && event.action.type === "error") check(event.action.error);
    });
    return () => { unsubscribeQueries(); unsubscribeMutations(); };
  }, [client]);
}

// ------------------------------------------------------------------ 428 : popup de risques puis relance explicite
/** Libellé de l'action interrompue, déduit de la requête d'origine. */
const ACTION_LABELS: [RegExp, string][] = [
  [/^\/runs$/, "Lancer l'analyse"],
  [/^\/applications\/[^/]+\/apply$/, "Lancer la candidature"],
  [/^\/applications\/[^/]+\/contacts\/lookup$/, "Chercher les recruteurs de l'offre"],
  [/^\/applications\/[^/]+\/messages$/, "Relire le message"],
  [/^\/profile\/import-linkedin$/, "Importer depuis LinkedIn"],
];
const actionLabel = (request: ApiRequest) => ACTION_LABELS.find(([re]) => re.test(request.path))?.[1] ?? "l'action interrompue";
/** Envoi d'un message : jamais renvoyé depuis le bandeau (I-04), seule la relecture permet d'envoyer. */
const MESSAGE_PATH = /^\/applications\/([^/]+)\/messages$/;

/**
 * Champs d'un message interrompu par un 428, gardés en mémoire pour rouvrir la relecture. `confirm` n'est jamais
 * repris : seul un nouveau clic sur « Confirmer l'envoi », case de relecture cochée, envoie (I-04).
 */
function composeInitialOf(body: unknown): ComposeInitial | null {
  if (!body || typeof body !== "object") return null;
  const b = body as Record<string, unknown>;
  if ((b.channel !== "email" && b.channel !== "linkedin") || typeof b.to !== "string" || typeof b.body !== "string") return null;
  return { channel: b.channel, to: b.to, subject: typeof b.subject === "string" ? b.subject : "", body: b.body, attach_cv: b.attach_cv !== false };
}

/** Relecture d'un message rouverte après acceptation des risques : rien n'est envoyé sans clic. */
function MessageReview({ jobId, initial, onClose }: { jobId: string; initial: ComposeInitial; onClose: () => void }) {
  const { data: app, error } = useApplication(jobId);
  if (error) return <div className="mb-4"><ErrorBox error={error} /></div>;
  if (!app) return null;
  const contact = app.contacts.find((c) => initial.to && (c.email === initial.to || c.profile_url === initial.to));
  return <ComposeModal app={app} contact={contact} initial={initial} onClose={onClose} />;
}

const isTask = (value: unknown): value is Task =>
  !!value && typeof value === "object" && "id" in value && "status" in value && "kind" in value;

/**
 * Prudence passée à « prudent » dans la popup : un `POST /runs` relancé l'est en `caution: "prudent"`, sans un
 * `max_items` supérieur au plafond « prudent » (le serveur le refuserait). Les autres requêtes sont inchangées.
 */
function withPrudentRun(request: ApiRequest): ApiRequest {
  if (request.path !== "/runs" || !request.body || typeof request.body !== "object") return request;
  const { max_items, ...rest } = request.body as Record<string, unknown>;
  const keep = max_items === undefined || (typeof max_items === "number" && max_items > CAUTION_MAX.prudent) ? {} : { max_items };
  return { ...request, body: { ...rest, ...keep, caution: "prudent" } };
}

/**
 * Sur un 428 (consentement absent ou d'une autre version, I-16) : popup de la source indiquée. Après acceptation,
 * AUCUN rejeu automatique (décision du 2026-10-09) : un bandeau « Relancer : {action} » attend un clic explicite.
 * Pour un message, la relance rouvre directement la relecture (champs gardés en mémoire) et n'envoie rien.
 */
function ConsentGate() {
  const { data: sources } = useSources();
  const { track } = useTaskUi();
  const navigate = useNavigate();
  const [pending, setPending] = useState<ApiError | null>(null);
  const [relaunch, setRelaunch] = useState<{ label: string; request: ApiRequest } | null>(null);
  const [relaunchError, setRelaunchError] = useState<unknown>(null);
  const [relaunching, setRelaunching] = useState(false);
  const [reviewing, setReviewing] = useState<{ jobId: string; initial: ComposeInitial } | null>(null);

  useEffect(() => onConsentRequired((error) => setPending((current) => current ?? error)), []);

  const code = pending?.source as SourceCode | undefined;
  const known = sources?.find((s) => s.code === code);
  // Liste des sources pas encore chargée : popup construite depuis le 428 (source et version).
  const source: SourceInfo | null = !pending || !code ? null : known && known.risk_version === (pending.riskVersion ?? known.risk_version)
    ? known
    : {
      code, label: known?.label ?? SOURCE_LABEL[code] ?? code, access_mode: "navigateur", enabled: true,
      risk_version: pending.riskVersion, consent: known?.consent ?? null, consent_required: true, blocked: null,
    };

  const onAccepted = ({ switchedToPrudent }: { switchedToPrudent: boolean }) => {
    if (pending?.request) {
      const request = switchedToPrudent ? withPrudentRun(pending.request) : pending.request;
      setRelaunch({ label: actionLabel(request), request });
    }
    setRelaunchError(null);
    setPending(null);
  };

  const run = async () => {
    if (!relaunch) return;
    const message = MESSAGE_PATH.exec(relaunch.request.path);
    if (message) {
      // I-04 : aucune requête d'envoi ; la relecture se rouvre, case et « Confirmer l'envoi » restent à faire.
      setRelaunch(null);
      const initial = composeInitialOf(relaunch.request.body);
      if (initial) setReviewing({ jobId: decodeURIComponent(message[1]), initial });
      else navigate(`/candidatures/${message[1]}`);
      return;
    }
    setRelaunching(true);
    setRelaunchError(null);
    try {
      const result = await api<unknown>(relaunch.request.path, { method: relaunch.request.method, body: relaunch.request.body });
      setRelaunch(null);
      if (isTask(result)) track(result);
    } catch (e) {
      setRelaunchError(e);
    } finally {
      setRelaunching(false);
    }
  };

  const isMessage = relaunch ? MESSAGE_PATH.test(relaunch.request.path) : false;
  return (
    <>
      {source && <RiskConsentDialog source={source} onAccepted={onAccepted} onRefused={() => setPending(null)} />}
      {reviewing && <MessageReview jobId={reviewing.jobId} initial={reviewing.initial} onClose={() => setReviewing(null)} />}
      {relaunch && (
        <div className="mb-4 space-y-2">
          <Notice tone="info">
            <span className="flex flex-wrap items-center gap-2">
              <span>
                {isMessage
                  ? "Risques acceptés. Rien n'a été envoyé : rouvre la relecture du message pour le confirmer."
                  : "Risques acceptés. L'action interrompue n'a pas été relancée."}
              </span>
              <Button small variant="primary" loading={relaunching} onClick={run}>Relancer : {relaunch.label}</Button>
              <Button small variant="ghost" onClick={() => setRelaunch(null)}>Ignorer</Button>
            </span>
          </Notice>
          <ErrorBox error={relaunchError} />
        </div>
      )}
    </>
  );
}

// ------------------------------------------------------------------ repos par source (I-06)
/** Bandeau de blocage par source, d'après `GET /api/sources` ; repli sur `/api/status.blocked` (LinkedIn). */
function BlockBanners() {
  const { data: status } = useStatus();
  const { data: sources, isError } = useSources();
  const clear = useClearBlock();
  const legacyClear = useAction(() => api("/block/clear", { method: "POST" }), [["status"]]);
  const blocked = sources
    ? sources.filter((s) => s.blocked).map((s) => ({ code: s.code as SourceCode | null, label: s.label, blocked: s.blocked! }))
    : isError && status?.blocked ? [{ code: null, label: "LinkedIn", blocked: status.blocked }] : [];
  return (
    <>
      {blocked.map(({ code, label, blocked: b }) => (
        <div key={code ?? "legacy"} className="mb-4">
          <Notice tone="danger">
            {label} a interrompu la navigation le {formatDate(b.at)} ({b.where}). Les actions qui pilotent le navigateur avec{" "}
            {label} sont suspendues jusqu'au {formatDate(b.until)}. Vérifie ton compte dans un navigateur normal, puis lève le
            blocage.{" "}
            <Button small variant="danger" loading={code ? clear.isPending : legacyClear.isPending}
              onClick={() => (code ? clear.mutate(code) : legacyClear.mutate(undefined))}>Lever le blocage</Button>
          </Notice>
        </div>
      ))}
      <ErrorBox error={clear.error ?? legacyClear.error} />
    </>
  );
}

export default function App() {
  const { data: status } = useStatus();
  const [dark, setDark] = useState(() => document.documentElement.dataset.theme === "dark");
  const toggleTheme = () => {
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("lm-theme", next); } catch { /* sans conséquence */ }
    setDark(!dark);
  };
  useRouteAnnouncements();
  useWatchUnauthorized();
  const tokenExpired = useTokenExpired();
  const { pathname } = useLocation();
  return (
    <div className="flex min-h-screen flex-col md:flex-row">
      <a
        href="#contenu"
        className="sr-only z-50 rounded-md bg-surface px-3 py-2 text-sm font-medium text-accent-primary shadow-lift focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Aller au contenu
      </a>
      <nav aria-label="Navigation" className="flex shrink-0 gap-1 overflow-x-auto border-b border-border-default bg-surface p-3 md:w-56 md:flex-col md:border-b-0 md:border-r">
        <div className="hidden px-3 pb-3 pt-1 md:block">
          <div className="text-sm font-semibold">LinkedIn Market</div>
          <div className="text-xs text-text-muted">Gestion des candidatures</div>
        </div>
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) =>
              `whitespace-nowrap rounded-md px-3 py-2 text-sm ${isActive ? "bg-accent-soft font-medium text-accent-primary" : "text-text-secondary hover:bg-bg-secondary"}`}
          >
            {item.label}
          </NavLink>
        ))}
        <button type="button" onClick={toggleTheme} aria-pressed={dark} className="ml-auto rounded-md px-3 py-2 text-left text-sm text-text-muted hover:bg-bg-secondary md:ml-0 md:mt-auto">
          Thème clair / sombre
        </button>
      </nav>
      <main id="contenu" tabIndex={-1} className="min-w-0 flex-1 p-4 md:p-6 focus:outline-none">
        <BlockBanners />
        <ConsentGate />
        {status && !status.credentials_configured && (
          <div className="mb-4">
            <Notice tone="warning">
              Identifiants LinkedIn absents : renseigne LINKEDIN_EMAIL et LINKEDIN_PASSWORD dans le fichier .env pour lancer
              une collecte, postuler ou envoyer des messages depuis l'interface.
            </Notice>
          </div>
        )}
        {tokenExpired && (
          <div className="mb-4">
            <Notice tone="danger">
              Le jeton local a changé : recharge la page.{" "}
              <button type="button" className="font-medium underline" onClick={() => window.location.reload()}>Recharger</button>
            </Notice>
          </div>
        )}
        <ErrorBoundary key={pathname}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/offres" element={<Jobs />} />
            <Route path="/candidatures" element={<Applications />} />
            <Route path="/candidatures/:jobId" element={<Applications />} />
            <Route path="/marche" element={<Suspense fallback={<div role="status" className="h-40 animate-pulse rounded-xl bg-bg-secondary"><span className="sr-only">Chargement du marché…</span></div>}><Market /></Suspense>} />
            <Route path="/competences" element={<Skills />} />
            <Route path="/parametres" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </main>
    </div>
  );
}
