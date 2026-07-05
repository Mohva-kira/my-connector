import React, { useCallback, useEffect, useState } from "react";
import { apiFetch, getAccessToken, setAccessToken } from "./apiClient";

type TenantRow = {
  id: string;
  name: string;
  slug: string;
  role: string;
  status: string;
};

export type Me = {
  email: string;
  active_tenant_id: string;
  tenants: TenantRow[];
};

type PlanRow = {
  id: string;
  slug: string;
  name: string;
  price_amount: number;
  currency: string;
  interval: string;
  quota_analyses_per_month: number | null;
};

async function parseJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text.trim()) return {};
  return JSON.parse(text) as unknown;
}

export function useSaasSession(saasEnabled: boolean, sessionTick: number) {
  const [me, setMe] = useState<Me | null>(null);

  useEffect(() => {
    if (!saasEnabled || !getAccessToken()) {
      setMe(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const res = await apiFetch("/api/auth/me");
        const data = (await parseJson(res)) as Me;
        if (!res.ok) throw new Error("session");
        if (!cancelled) setMe(data);
      } catch {
        if (!cancelled) setMe(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [saasEnabled, sessionTick]);

  const switchTenant = useCallback(async (tenantId: string) => {
    const res = await apiFetch("/api/auth/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ tenant_id: tenantId }),
    });
    const data = (await parseJson(res)) as { access_token?: string; detail?: unknown };
    if (!res.ok || !data.access_token) {
      throw new Error(typeof data.detail === "string" ? data.detail : "switch failed");
    }
    setAccessToken(data.access_token);
    const meRes = await apiFetch("/api/auth/me");
    const meData = (await parseJson(meRes)) as Me;
    if (meRes.ok) setMe(meData);
  }, []);

  return { me, setMe, switchTenant };
}

export function ImapSettingsForm({
  tenantId,
  onSaved,
  showCancel,
  onCancel,
  titleId,
}: {
  tenantId: string;
  onSaved: () => void;
  showCancel?: boolean;
  onCancel?: () => void;
  /** Pour aria-labelledby en modale */
  titleId?: string;
}) {
  const [host, setHost] = useState("");
  const [port, setPort] = useState(993);
  const [user, setUser] = useState("");
  const [password, setPassword] = useState("");
  const [folder, setFolder] = useState("INBOX");
  const [ssl, setSsl] = useState(true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function save(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    setBusy(true);
    try {
      const body: Record<string, unknown> = {
        imap_host: host.trim(),
        imap_port: port,
        imap_user: user.trim(),
        imap_folder: folder.trim() || "INBOX",
        imap_use_ssl: ssl,
      };
      if (password.trim()) body.imap_password = password;
      const res = await apiFetch(`/api/tenants/${tenantId}/imap`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await parseJson(res);
      if (!res.ok) {
        throw new Error(
          data && typeof data === "object" && "detail" in data
            ? String((data as { detail: unknown }).detail)
            : "Erreur",
        );
      }
      onSaved();
      if (showCancel && onCancel) onCancel();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Erreur");
    } finally {
      setBusy(false);
    }
  }

  async function testConn() {
    setMsg(null);
    setBusy(true);
    try {
      const res = await apiFetch(`/api/tenants/${tenantId}/imap/test`, { method: "POST" });
      const data = await parseJson(res);
      if (!res.ok) {
        throw new Error(
          data && typeof data === "object" && "detail" in data
            ? String((data as { detail: unknown }).detail)
            : "Échec",
        );
      }
      setMsg("Connexion OK");
    } catch (err) {
      setMsg(err instanceof Error ? err.message : "Erreur");
    } finally {
      setBusy(false);
    }
  }

  const formProps = titleId
    ? { "aria-labelledby": titleId }
    : {};

  return (
    <form className="mt-4 space-y-3" onSubmit={(e) => void save(e)} {...formProps}>
      <label className="block text-sm">
        Hôte
        <input
          className="mt-1 w-full rounded-lg border border-stone-200 px-2 py-2 text-sm"
          value={host}
          onChange={(e) => setHost(e.target.value)}
          required
          placeholder="imap.example.com"
        />
      </label>
      <label className="block text-sm">
        Port
        <input
          type="number"
          className="mt-1 w-full rounded-lg border border-stone-200 px-2 py-2 text-sm"
          value={port}
          onChange={(e) => setPort(Number(e.target.value))}
        />
      </label>
      <label className="block text-sm">
        Utilisateur
        <input
          className="mt-1 w-full rounded-lg border border-stone-200 px-2 py-2 text-sm"
          value={user}
          onChange={(e) => setUser(e.target.value)}
          required
        />
      </label>
      <label className="block text-sm">
        Mot de passe
        <input
          type="password"
          className="mt-1 w-full rounded-lg border border-stone-200 px-2 py-2 text-sm"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="••••••••"
        />
      </label>
      <label className="block text-sm">
        Dossier
        <input
          className="mt-1 w-full rounded-lg border border-stone-200 px-2 py-2 text-sm"
          value={folder}
          onChange={(e) => setFolder(e.target.value)}
        />
      </label>
      <label className="flex items-center gap-2 text-sm">
        <input type="checkbox" checked={ssl} onChange={(e) => setSsl(e.target.checked)} />
        SSL (993)
      </label>
      {msg ? <p className="text-sm text-slate-700">{msg}</p> : null}
      <div className="flex flex-wrap gap-2 pt-2">
        <button
          type="submit"
          disabled={busy}
          className="rounded-xl bg-slate-800 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          Enregistrer
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => void testConn()}
          className="rounded-xl border border-stone-200 px-4 py-2 text-sm font-medium text-slate-800"
        >
          Tester
        </button>
        {showCancel && onCancel ? (
          <button type="button" onClick={onCancel} className="rounded-xl px-4 py-2 text-sm text-slate-600">
            Annuler
          </button>
        ) : null}
      </div>
    </form>
  );
}

export function ImapSettingsModal({
  tenantId,
  onClose,
  onSaved,
}: {
  tenantId: string;
  onClose: () => void;
  onSaved: () => void;
}) {
  const titleId = "imap-modal-title";
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" role="dialog" aria-modal="true" aria-labelledby={titleId}>
      <div className="max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-2xl bg-white p-6 shadow-xl">
        <h3 id={titleId} className="text-lg font-semibold text-slate-900">
          Boîte IMAP (organisation)
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Ces identifiants sont chiffrés côté serveur. Laissez le mot de passe vide pour ne pas le modifier.
        </p>
        <ImapSettingsForm
          tenantId={tenantId}
          onSaved={onSaved}
          showCancel
          onCancel={onClose}
          titleId={titleId}
        />
      </div>
    </div>
  );
}

export function BillingModal({
  onClose,
}: {
  onClose: () => void;
}) {
  const [plans, setPlans] = useState<PlanRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await apiFetch("/api/billing/plans");
        const data = (await parseJson(res)) as PlanRow[];
        if (!cancelled && res.ok) setPlans(Array.isArray(data) ? data : []);
      } catch {
        if (!cancelled) setErr("Impossible de charger les plans");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function pay(slug: string) {
    setErr(null);
    setBusy(slug);
    try {
      const res = await apiFetch("/api/billing/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan_slug: slug }),
      });
      const data = (await parseJson(res)) as { payment_url?: string; detail?: unknown };
      if (!res.ok || !data.payment_url) {
        throw new Error(typeof data.detail === "string" ? data.detail : "Checkout indisponible");
      }
      window.location.href = data.payment_url;
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Erreur");
    } finally {
      setBusy(null);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="billing-modal-title"
    >
      <div className="w-full max-w-lg rounded-2xl bg-white p-6 shadow-xl">
        <h3 id="billing-modal-title" className="text-lg font-semibold text-slate-900">
          Facturation CinetPay
        </h3>
        <p className="mt-1 text-sm text-slate-500">
          Vous serez redirigé vers le guichet de paiement sécurisé.
        </p>
        {err ? <p className="mt-2 text-sm text-red-700">{err}</p> : null}
        <ul className="mt-4 space-y-3">
          {plans.map((p) => (
            <li
              key={p.id}
              className="flex items-center justify-between rounded-xl border border-stone-200 px-4 py-3"
            >
              <div>
                <p className="font-medium text-slate-900">{p.name}</p>
                <p className="text-xs text-slate-500">
                  {p.price_amount} {p.currency} / {p.interval}
                  {p.quota_analyses_per_month != null ? ` — ${p.quota_analyses_per_month} analyses/mois` : ""}
                </p>
              </div>
              <button
                type="button"
                disabled={busy !== null}
                onClick={() => void pay(p.slug)}
                className="rounded-lg bg-emerald-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-800 disabled:opacity-50"
              >
                {busy === p.slug ? "…" : "Payer"}
              </button>
            </li>
          ))}
        </ul>
        <button type="button" className="mt-6 text-sm text-slate-600 underline" onClick={onClose}>
          Fermer
        </button>
      </div>
    </div>
  );
}
