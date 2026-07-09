import React, { useState } from "react";
import { apiFetch, setAccessToken } from "./apiClient";

type Mode = "login" | "register";

export default function AuthPanel({
  onSuccess,
}: {
  onSuccess: () => void;
}) {
  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [orgName, setOrgName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "register") {
        const res = await apiFetch("/api/auth/register", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            email: email.trim(),
            password,
            organization_name: orgName.trim(),
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data));
        }
        setAccessToken((data as { access_token: string }).access_token);
      } else {
        const res = await apiFetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: email.trim(), password }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data));
        }
        setAccessToken((data as { access_token: string }).access_token);
      }
      onSuccess();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erreur");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-md rounded-2xl border border-border-default bg-surface p-8 shadow-sm">
      <h2 className="text-center text-lg font-semibold text-text-primary">
        {mode === "login" ? "Connexion" : "Créer un compte"}
      </h2>
      <div className="mt-4 flex justify-center gap-2">
        <button
          type="button"
          className={`rounded-full px-4 py-1.5 text-sm font-medium ${
            mode === "login" ? "bg-accent-primary text-white" : "bg-bg-tertiary text-text-secondary"
          }`}
          onClick={() => setMode("login")}
        >
          Connexion
        </button>
        <button
          type="button"
          className={`rounded-full px-4 py-1.5 text-sm font-medium ${
            mode === "register" ? "bg-accent-primary text-white" : "bg-bg-tertiary text-text-secondary"
          }`}
          onClick={() => setMode("register")}
        >
          Inscription
        </button>
      </div>
      <form className="mt-6 space-y-4" onSubmit={(e) => void submit(e)}>
        <label className="block text-sm">
          <span className="text-text-secondary">Email</span>
          <input
            type="email"
            required
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mt-1 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2 text-sm text-text-primary"
          />
        </label>
        <label className="block text-sm">
          <span className="text-text-secondary">Mot de passe</span>
          <input
            type="password"
            required
            minLength={8}
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mt-1 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2 text-sm text-text-primary"
          />
        </label>
        {mode === "register" ? (
          <label className="block text-sm">
            <span className="text-text-secondary">Nom de l&apos;organisation</span>
            <input
              type="text"
              required
              value={orgName}
              onChange={(e) => setOrgName(e.target.value)}
              className="mt-1 w-full rounded-xl border border-border-default bg-bg-tertiary/60 px-3 py-2 text-sm text-text-primary"
            />
          </label>
        ) : null}
        {error ? (
          <p className="text-sm text-danger" role="alert">
            {error}
          </p>
        ) : null}
        <button
          type="submit"
          disabled={busy}
          className="w-full rounded-xl bg-accent-primary py-2.5 text-sm font-semibold text-white hover:bg-accent-hover disabled:opacity-50"
        >
          {busy ? "…" : mode === "login" ? "Se connecter" : "S&apos;inscrire"}
        </button>
      </form>
    </div>
  );
}
