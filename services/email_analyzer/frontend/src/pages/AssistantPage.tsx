import { Navigate, useSearchParams } from "react-router-dom";
import { getAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import PortfolioAssistant from "../PortfolioAssistant";
import { useSaasSession } from "../SaasPanels";

// Même composant que le panneau assistant de BriefPage — page dédiée en plein
// écran pour l'entrée de sidebar "Assistant" (utile sur les écrans où le
// Brief empile déjà stats + recommandations au-dessus).
export default function AssistantPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);
  const [searchParams] = useSearchParams();
  const initialQuestion = searchParams.get("q") ?? undefined;

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
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Assistant</h1>
      </header>
      <div className="mx-auto max-w-3xl">
        <PortfolioAssistant initialQuestion={initialQuestion} />
      </div>
    </AppShell>
  );
}
