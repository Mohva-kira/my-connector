import React from "react";
import { NavLink } from "react-router-dom";
import type { Me } from "../SaasPanels";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-full px-3 py-1.5 text-sm font-medium transition ${
    isActive ? "bg-accent-primary text-white" : "text-text-secondary hover:bg-bg-tertiary"
  }`;

export default function AppShell({
  me,
  children,
}: {
  me: Me;
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen px-4 py-8 sm:px-6 lg:py-10">
      <div className="mx-auto max-w-7xl">
        <nav
          className="mb-6 flex flex-wrap items-center justify-center gap-2 border-b border-border-default pb-4 text-sm lg:justify-between"
          aria-label="Navigation principale"
        >
          <div className="flex flex-wrap items-center justify-center gap-2">
            <NavLink to="/" end className={linkClass}>
              Accueil
            </NavLink>
            <NavLink to="/projects" className={linkClass}>
              Projets
            </NavLink>
            <NavLink to="/settings" className={linkClass}>
              Paramètres
            </NavLink>
          </div>
          <span className="text-text-muted">{me.email}</span>
        </nav>
        {children}
      </div>
    </div>
  );
}
