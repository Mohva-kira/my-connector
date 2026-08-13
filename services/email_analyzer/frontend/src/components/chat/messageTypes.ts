import type { ProjectAnalysis } from "../../analysisTypes";

export type AnalyzeProgress = { processed: number; total: number };

export type EmptyReportInfo = {
  message?: string;
  imapFolder?: string;
  daysBack?: number;
  widenDayOptions: number[];
};

export type HomeMessage =
  | { id: string; role: "user" | "assistant"; kind: "text"; content: string }
  | { id: string; role: "assistant"; kind: "status"; content: string; progress?: AnalyzeProgress }
  | { id: string; role: "assistant"; kind: "empty-report"; info: EmptyReportInfo }
  | { id: string; role: "assistant"; kind: "error"; content: string }
  | {
      id: string;
      role: "assistant";
      kind: "analysis";
      projectName: string;
      analysis: ProjectAnalysis;
      // Restreint au résultat d'une analyse sans filtre (voir
      // NO_FILTER_RESULT_KEY côté hook) — mêmes règles qu'avant la refonte
      // chat-first (AnalysisDashboard.onSelectThread).
      allowThreadDrilldown: boolean;
    };
