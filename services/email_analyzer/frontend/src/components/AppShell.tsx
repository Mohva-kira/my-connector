import React, { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import type { Me } from "../SaasPanels";

const NAV_ITEMS: Array<{ to: string; label: string; icon: string; end?: boolean }> = [
  { to: "/", label: "Brief", icon: "🏠", end: true },
  { to: "/projects", label: "Projets", icon: "📂" },
  { to: "/agenda", label: "Agenda", icon: "📅" },
  { to: "/actions", label: "Actions", icon: "✅" },
  { to: "/assistant", label: "Assistant", icon: "💬" },
  { to: "/settings", label: "Paramètres", icon: "⚙" },
];

// Sections mentionnées dans la vision produit mais sans donnée réelle
// derrière aujourd'hui (pas de parsing de pièces jointes, pas d'entité
// client/contact) — visibles mais désactivées plutôt qu'omises ou simulées.
const COMING_SOON_ITEMS: Array<{ label: string; icon: string }> = [
  { label: "Documents", icon: "📄" },
  { label: "Clients", icon: "👥" },
];

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2.5 rounded-xl px-3 py-2 text-sm font-medium transition ${
    isActive
      ? "bg-accent-soft text-accent-primary"
      : "text-text-secondary hover:bg-bg-tertiary hover:text-text-primary"
  }`;

// Barre de recherche en langage naturel : pas de moteur séparé — envoie la
// question à l'assistant portefeuille (même LLM, même contexte projets) via
// /assistant?q=…, lu par AssistantPage/PortfolioAssistant pour l'envoyer dès
// le montage. Accessible depuis n'importe quelle page (sidebar partagée),
// pas seulement depuis le panneau assistant du Brief.
function SidebarSearch() {
  const [query, setQuery] = useState("");
  const navigate = useNavigate();

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    navigate(`/assistant?q=${encodeURIComponent(trimmed)}`);
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

export default function AppShell({
  me,
  children,
}: {
  me: Me;
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-bg-primary lg:flex">
      <aside
        className="border-b border-border-default bg-surface px-4 py-4 lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-60 lg:flex-none lg:flex-col lg:border-b-0 lg:border-r lg:px-3 lg:py-6"
        aria-label="Navigation principale"
      >
        <div className="mb-3 flex items-center justify-between lg:mb-5 lg:block lg:px-2">
          <p className="text-sm font-semibold text-text-primary">my-connector</p>
          <p className="truncate text-xs text-text-muted lg:hidden">{me.email}</p>
        </div>
        <SidebarSearch />
        <nav className="flex flex-wrap gap-1.5 lg:flex-col lg:gap-1">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} className={linkClass}>
              <span aria-hidden>{item.icon}</span>
              {item.label}
            </NavLink>
          ))}
          {COMING_SOON_ITEMS.map((item) => (
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
          ))}
        </nav>
        <div className="mt-auto hidden pt-4 lg:block">
          <p className="truncate text-xs text-text-muted">{me.email}</p>
        </div>
      </aside>
      <main className="flex-1 px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
        <div className="mx-auto max-w-7xl">{children}</div>
      </main>
    </div>
  );
}
