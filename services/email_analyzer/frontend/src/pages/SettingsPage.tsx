import React, { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { getAccessToken, setAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import ConnectedAccounts from "../ConnectedAccounts";
import { BillingModal, ImapSettingsForm, useSaasSession } from "../SaasPanels";

export default function SettingsPage({
  healthError,
  saasEnabled,
  billingReturnBanner,
  sessionTick,
  setSessionTick,
}: {
  healthError: string | null;
  saasEnabled: boolean;
  billingReturnBanner: boolean;
  sessionTick: number;
  setSessionTick: React.Dispatch<React.SetStateAction<number>>;
}) {
  const navigate = useNavigate();
  const [showBillingModal, setShowBillingModal] = useState(false);
  const { me, switchTenant } = useSaasSession(saasEnabled, sessionTick);

  if (!saasEnabled) {
    return <Navigate to="/" replace />;
  }

  if (!getAccessToken()) {
    return <Navigate to="/login" replace />;
  }

  if (!me) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-bg-primary px-4">
        <p className="text-sm text-text-muted">Chargement du compte…</p>
      </div>
    );
  }

  return (
    <AppShell me={me}>
      {billingReturnBanner ? (
        <div
          className="mb-6 rounded-2xl border border-success/30 bg-success/10 px-4 py-3 text-center text-sm text-success"
          role="status"
        >
          Retour depuis le paiement : si le montant est validé, votre organisation sera activée sous peu
          (notification serveur CinetPay).
        </div>
      ) : null}

      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Paramètres</h1>
        <p className="mt-1 text-sm text-text-muted">Compte, organisation, boîte IMAP et facturation.</p>
      </header>

      {healthError ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger"
        >
          Impossible de joindre l&apos;API ({healthError}). Vérifiez que{" "}
          <code className="text-xs">uvicorn</code> tourne sur le port 8000.
        </div>
      ) : null}

      <div className="mx-auto max-w-2xl space-y-6">
        <section className="rounded-2xl border border-border-default bg-surface p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-text-primary">Compte et organisation</h2>
          <p className="mt-1 text-sm text-text-muted">Email connecté et organisation active.</p>
          <dl className="mt-4 space-y-3 text-sm">
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-text-muted">Email</dt>
              <dd className="mt-0.5 text-text-primary">{me.email}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-text-muted">Organisation</dt>
              <dd className="mt-2">
                {me.tenants.length > 1 ? (
                  <select
                    className="w-full max-w-md rounded-lg border border-border-default bg-bg-tertiary/60 px-2 py-2 text-text-primary"
                    value={me.active_tenant_id}
                    onChange={(e) => void switchTenant(e.target.value)}
                    title="Organisation active"
                  >
                    {me.tenants.map((t) => (
                      <option key={t.id} value={t.id}>
                        {t.name} ({t.status})
                      </option>
                    ))}
                  </select>
                ) : (
                  <span className="inline-block rounded-full bg-bg-tertiary px-3 py-1 text-xs text-text-secondary">
                    {me.tenants[0]?.name ?? "—"} ({me.tenants[0]?.status ?? "—"})
                  </span>
                )}
              </dd>
            </div>
          </dl>
        </section>

        <section className="rounded-2xl border border-border-default bg-surface p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-text-primary">Boîte IMAP</h2>
          <p className="mt-1 text-sm text-text-muted">
            Ces identifiants sont chiffrés côté serveur. Laissez le mot de passe vide pour ne pas le modifier.
          </p>
          <ImapSettingsForm
            tenantId={me.active_tenant_id}
            onSaved={() => setSessionTick((s) => s + 1)}
          />
        </section>

        <ConnectedAccounts />

        <section className="rounded-2xl border border-border-default bg-surface p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-text-primary">Facturation</h2>
          <p className="mt-1 text-sm text-text-muted">Plans et paiement CinetPay.</p>
          <button
            type="button"
            className="mt-4 rounded-full bg-success px-4 py-2 text-sm font-medium text-white shadow hover:opacity-90"
            onClick={() => setShowBillingModal(true)}
          >
            Ouvrir la facturation
          </button>
        </section>

        <section className="rounded-2xl border border-border-default bg-surface p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-text-primary">Session</h2>
          <p className="mt-1 text-sm text-text-muted">Quitter l&apos;application sur cet appareil.</p>
          <button
            type="button"
            className="mt-4 text-sm font-medium text-text-secondary underline decoration-border-default underline-offset-2 hover:text-text-primary"
            onClick={() => {
              setAccessToken(null);
              setSessionTick((s) => s + 1);
              navigate("/login", { replace: true });
            }}
          >
            Déconnexion
          </button>
        </section>
      </div>

      {showBillingModal ? <BillingModal onClose={() => setShowBillingModal(false)} /> : null}
    </AppShell>
  );
}
