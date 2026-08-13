import React, { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import type { Me } from "../SaasPanels";
import { useActiveJobIndicator } from "../hooks/useActiveJobIndicator";
import ThemeToggle from "./ThemeToggle";

const SAAS_NAV_ITEMS: Array<{ to: string; label: string; icon: string; end?: boolean }> = [
  { to: "/", label: "Accueil", icon: "🏠", end: true },
  { to: "/projects", label: "Projets", icon: "📂" },
  { to: "/agenda", label: "Agenda", icon: "📅" },
  { to: "/actions", label: "Actions", icon: "✅" },
  { to: "/memory", label: "Mémoire", icon: "🧠" },
  { to: "/objectives", label: "Objectifs", icon: "🎯" },
  { to: "/documents", label: "Documents", icon: "📄" },
  { to: "/settings", label: "Paramètres", icon: "⚙" },
];

// En mode legacy (.env, sans DB), /projects, /agenda, /actions et /settings
// redirigent tous vers / (voir leurs gardes `Navigate` respectives) — le
// rail n'affiche donc que l'accueil, pour ne pas montrer de liens morts.
const LEGACY_NAV_ITEMS: Array<{ to: string; label: string; icon: string; end?: boolean }> = [
  { to: "/", label: "Accueil", icon: "🏠", end: true },
];

// Sections mentionnées dans la vision produit ("25 domaines de contexte",
// context/project-overview.md) mais sans intégration externe réelle
// aujourd'hui — visibles mais désactivées plutôt qu'omises ou simulées.
// Regroupées par thème plutôt qu'une entrée par domaine (25 entrées grisées
// nuirait à la lisibilité de la nav sans servir la compréhension du
// travail de l'utilisateur — voir project-overview.md "Principes de
// développement") ; le détail domaine par domaine (statut, prérequis) vit
// dans project-overview.md, pas ici.
const COMING_SOON_ITEMS: Array<{ label: string; icon: string }> = [
  { label: "Clients", icon: "👥" },
  { label: "Tâches", icon: "🗒" },
  { label: "Communication", icon: "💬" },
  { label: "Finance", icon: "💳" },
  { label: "Santé", icon: "🩺" },
  { label: "Développement", icon: "🛠" },
  { label: "Automatisation", icon: "🔁" },
];

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium transition ${
    isActive
      ? "bg-accent-soft text-accent-primary"
      : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
  }`;

// Barre de recherche en langage naturel : pas de moteur séparé — envoie la
// question à l'assistant portefeuille (même LLM, même contexte projets) via
// /?q=…, lu par ChatHome (mode SaaS) pour l'envoyer dès le montage.
// Accessible depuis n'importe quelle page (sidebar partagée), pas
// seulement depuis l'accueil. Uniquement affichée en mode SaaS : en
// legacy, ChatHome n'a pas de branchement `?q=` (pas de portfolio
// assistant sans tenant).
function SidebarSearch() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    navigate(`/?q=${encodeURIComponent(trimmed)}`);
    setQuery("");
  }

  return (
    <form onSubmit={handleSubmit} className="mb-3 lg:mb-4">
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Que recherchez-vous ?"
        aria-label="Recherche en langage naturel"
        className="w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2 text-sm text-text-primary outline-none ring-accent-primary placeholder:text-text-muted focus:border-accent-primary focus:bg-bg-tertiary focus:ring-2"
      />
    </form>
  );
}

// Badge discret : une analyse (manuelle, cron ou auto-sync au login) tourne
// en ce moment pour ce tenant. Simple présence (voir useActiveJobIndicator),
// pas une barre de progression — celle-ci reste dans ChatThread pour le job
// que l'utilisateur vient lui-même de déclencher.
function AnalysisBadge() {
  return (
    <span
      role="status"
      className="flex items-center gap-1.5 rounded-full bg-accent-soft px-2.5 py-1 text-xs font-medium text-accent-primary"
      title="Une analyse est en cours en arrière-plan">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent-primary" aria-hidden />
      Analyse en cours…
    </span>
  );
}

export default function AppShell({
  me,
  saasEnabled = true,
  children,
}: {
  me: Me | null;
  // Optionnel, défaut true : tous les appelants actuels (Projets, Agenda,
  // Actions, Paramètres, Brief) ne rendent AppShell qu'en mode SaaS. Le
  // futur accueil chat-first (ChatHome) sera le premier à passer `false`
  // en mode legacy.
  saasEnabled?: boolean;
  children: React.ReactNode;
}) {
  const navItems = saasEnabled ? SAAS_NAV_ITEMS : LEGACY_NAV_ITEMS;
  const analysisActive = useActiveJobIndicator(saasEnabled);

  return (
    <div className="min-h-screen bg-bg-primary lg:flex">
      <aside
        className="border-b border-border-default bg-surface px-4 py-4 lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-60 lg:flex-none lg:flex-col lg:border-b-0 lg:border-r lg:px-3 lg:py-6"
        aria-label="Navigation principale"
      >
        <div className="mb-3 flex items-center justify-between lg:mb-5 lg:px-2">
          <p className="text-sm font-semibold text-text-primary">my-connector</p>
          <div className="flex items-center gap-2 lg:hidden">
            {analysisActive ? <AnalysisBadge /> : null}
            {me ? <p className="truncate text-xs text-text-muted">{me.email}</p> : null}
            <ThemeToggle />
          </div>
        </div>
        {saasEnabled ? <SidebarSearch /> : null}
        <nav className="flex flex-wrap gap-1.5 lg:flex-col lg:gap-1">
          {navItems.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={linkClass}>
              <span aria-hidden>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
          {saasEnabled
            ? COMING_SOON_ITEMS.map((item) => (
                <span
                  key={item.label}
                  className="flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium text-text-muted"
                  title="Bientôt disponible"
                >
                  <span aria-hidden>{item.icon}</span>
                  {item.label}
                  <span className="ml-auto rounded-full bg-bg-tertiary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    Bientôt
                  </span>
                </span>
              ))
            : null}
        </nav>
        <div className="mt-auto hidden flex-col gap-2 pt-4 lg:flex lg:px-2">
          {analysisActive ? <AnalysisBadge /> : null}
          <div className="flex items-center justify-between gap-2">
            {me ? <p className="truncate text-xs text-text-muted">{me.email}</p> : <span />}
            <ThemeToggle />
          </div>
        </div>
      </aside>
      <main className="flex-1 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
        <div className="mx-auto max-w-7xl">{children}</div>
      </main>
    </div>
  );
}
