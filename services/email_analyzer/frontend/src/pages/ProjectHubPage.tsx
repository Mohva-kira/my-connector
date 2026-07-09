import { Navigate } from "react-router-dom";
import { getAccessToken } from "../apiClient";
import AppShell from "../components/AppShell";
import ProjectHub from "../ProjectHub";
import { useSaasSession } from "../SaasPanels";

// Project Hub (architecture.md) nécessite le mode SaaS : les projets sont
// scopés par tenant (Unit 10/11), pas de version legacy sans DB.
export default function ProjectHubPage({
  saasEnabled,
  sessionTick,
}: {
  saasEnabled: boolean;
  sessionTick: number;
}) {
  const { me } = useSaasSession(saasEnabled, sessionTick);

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
      <header className="mb-8 text-center lg:mb-10">
        <h1 className="text-2xl font-semibold tracking-tight text-text-primary">Projets</h1>
      </header>
      <ProjectHub />
    </AppShell>
  );
}
