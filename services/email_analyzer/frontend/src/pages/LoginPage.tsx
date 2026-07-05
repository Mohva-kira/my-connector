import React from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import AuthPanel from "../AuthPanel";
import { getAccessToken } from "../apiClient";

export default function LoginPage({
  saasEnabled,
  healthError,
  onAuthed,
}: {
  saasEnabled: boolean;
  healthError: string | null;
  onAuthed: () => void;
}) {
  const navigate = useNavigate();

  if (saasEnabled && getAccessToken()) {
    return <Navigate to="/" replace />;
  }

  if (saasEnabled) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-stone-100/80 px-4 py-12">
        <AuthPanel
          onSuccess={() => {
            onAuthed();
            navigate("/", { replace: true });
          }}
        />
      </div>
    );
  }

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-stone-100/80 px-4 py-12">
      <div className="mx-auto w-full max-w-md rounded-2xl border border-stone-200 bg-white p-8 shadow-sm">
        <h2 className="text-center text-lg font-semibold text-slate-900">Connexion</h2>
        {healthError ? (
          <p className="mt-4 text-sm leading-relaxed text-slate-600">
            Impossible de joindre l&apos;API ({healthError}). Démarrez le serveur (ex.{" "}
            <code className="rounded bg-stone-100 px-1 text-xs">uvicorn</code> sur le port 8000) puis rechargez cette page.
          </p>
        ) : (
          <p className="mt-4 text-sm leading-relaxed text-slate-600">
            L&apos;authentification par compte n&apos;est disponible que lorsque le backend est en{" "}
            <strong className="font-medium text-slate-800">mode SaaS</strong>. En mode analyse locale, les identifiants IMAP
            sont lus depuis le fichier <code className="rounded bg-stone-100 px-1 text-xs">.env</code> du dépôt — utilisez
            directement l&apos;accueil pour lancer une analyse.
          </p>
        )}
        <Link
          to="/"
          className="mt-6 inline-flex w-full items-center justify-center rounded-xl bg-slate-800 py-2.5 text-sm font-semibold text-white hover:bg-slate-900"
        >
          Retour à l&apos;accueil
        </Link>
      </div>
    </div>
  );
}
