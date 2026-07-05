import React, { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { getAccessToken, setAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
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
      <div className="flex min-h-screen flex-col items-center justify-center bg-stone-100/80 px-4">
        <p className="text-sm text-slate-500">Chargement du compte…</p>
      </div>
    );
  }

  return (
    <AppShell me={me}>
      {billingReturnBanner ? (
        <div
          className="mb-6 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-center text-sm text-emerald-900"
          role="status"
        >
          Retour depuis le paiement : si le montant est validé, votre organisation sera activée sous peu
          (notification serveur CinetPay).
        </div>
      ) : null}

      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-slate-900">Paramètres</h1>
        <p className="mt-1 text-sm text-slate-500">Compte, organisation, boîte IMAP et facturation.</p>
      </header>

      {healthError ? (
        <div
          role="alert"
          className="mb-6 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-900"
        >
          Impossible de joindre l&apos;API ({healthError}). Vérifiez que{" "}
          <code className="text-xs">uvicorn</code> tourne sur le port 8000.
        </div>
      ) : null}

      <div className="mx-auto max-w-2xl space-y-6">
        <section className="rounded-2xl border border-stone-200/80 bg-white p-6 shadow-sm shadow-stone-200/40">
          <h2 className="text-lg font-semibold text-slate-900">Compte et organisation</h2>
          <p className="mt-1 text-sm text-slate-500">Email connecté et organisation active.</p>
          <dl className="mt-4 space-y-3 text-sm">
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">Email</dt>
              <dd className="mt-0.5 text-slate-800">{me.email}</dd>
            </div>
            <div>
              <dt className="text-xs font-medium uppercase tracking-wide text-slate-400">Organisation</dt>
              <dd className="mt-2">
                {me.tenants.length > 1 ? (
                  <select
                    className="w-full max-w-md rounded-lg border border-stone-200 px-2 py-2 text-slate-800"
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
                  <span className="inline-block rounded-full bg-stone-100 px-3 py-1 text-xs text-slate-600">
                    {me.tenants[0]?.name ?? "—"} ({me.tenants[0]?.status ?? "—"})
                  </span>
                )}
              </dd>
            </div>
          </dl>
        </section>

        <section className="rounded-2xl border border-stone-200/80 bg-white p-6 shadow-sm shadow-stone-200/40">
          <h2 className="text-lg font-semibold text-slate-900">Boîte IMAP</h2>
          <p className="mt-1 text-sm text-slate-500">
            Ces identifiants sont chiffrés côté serveur. Laissez le mot de passe vide pour ne pas le modifier.
          </p>
          <ImapSettingsForm
            tenantId={me.active_tenant_id}
            onSaved={() => setSessionTick((s) => s + 1)}
          />
        </section>

        <section className="rounded-2xl border border-stone-200/80 bg-white p-6 shadow-sm shadow-stone-200/40">
          <h2 className="text-lg font-semibold text-slate-900">Facturation</h2>
          <p className="mt-1 text-sm text-slate-500">Plans et paiement CinetPay.</p>
          <button
            type="button"
            className="mt-4 rounded-full bg-emerald-700 px-4 py-2 text-sm font-medium text-white shadow hover:bg-emerald-800"
            onClick={() => setShowBillingModal(true)}
          >
            Ouvrir la facturation
          </button>
        </section>

        <section className="rounded-2xl border border-stone-200/80 bg-white p-6 shadow-sm shadow-stone-200/40">
          <h2 className="text-lg font-semibold text-slate-900">Session</h2>
          <p className="mt-1 text-sm text-slate-500">Quitter l&apos;application sur cet appareil.</p>
          <button
            type="button"
            className="mt-4 text-sm font-medium text-slate-600 underline decoration-slate-300 underline-offset-2 hover:text-slate-900"
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
