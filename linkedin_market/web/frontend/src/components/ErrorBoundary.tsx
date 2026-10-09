import { Component, ErrorInfo, ReactNode } from "react";
import { Link } from "react-router-dom";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

/**
 * Frontière d'erreur de rendu (frontend/03 § 3.3) : une donnée inattendue ne vide plus toute l'application. Placée
 * autour des routes ; le tiroir des tâches reste en dehors pour rester utilisable. Remontée (clé) à chaque page.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Erreur de rendu", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div role="alert" className="rounded-lg border border-danger/40 bg-surface p-6">
        <h1 tabIndex={-1} className="text-xl font-semibold focus:outline-none">Cette page a rencontré une erreur</h1>
        <p className="mt-2 text-sm text-text-secondary">
          Une donnée inattendue a empêché l'affichage. Le reste de l'application fonctionne : les tâches restent accessibles.
        </p>
        <pre className="mt-3 max-h-32 overflow-auto whitespace-pre-wrap rounded-md bg-bg-secondary p-3 text-xs text-text-secondary">{this.state.error.message}</pre>
        <div className="mt-4 flex flex-wrap gap-2">
          <Link to="/" className="rounded-md bg-accent-primary px-3.5 py-2 text-sm font-medium text-on-accent hover:bg-accent-hover">
            Revenir au tableau de bord
          </Link>
          <button type="button" onClick={() => this.setState({ error: null })}
            className="rounded-md border border-border-default bg-surface px-3.5 py-2 text-sm font-medium hover:bg-bg-secondary">
            Réessayer
          </button>
        </div>
      </div>
    );
  }
}
