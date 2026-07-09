/**
 * Pont Node.js vers l'analyse Python (`python -m email_analyzer`).
 * Prérequis : PYTHONPATH ou exécution depuis la racine du dépôt avec le chemin ci-dessous.
 */

import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** Racine du dépôt (my-connector) : bridge/ → email_analyzer/ → services/ → racine */
export function getRepoRoot() {
  return path.resolve(__dirname, "..", "..", "..");
}

/**
 * Lance l'analyse pour un projet et retourne le rapport JSON parsé.
 * @param {string} projectName - Filtre projet (même sémantique que le CLI Python)
 * @param {{ period?: string, days?: number, pythonPath?: string, timeoutMs?: number }} [options]
 * @returns {Promise<Record<string, unknown>>}
 */
export function runEmailAnalyzer(projectName, options = {}) {
  const repoRoot = getRepoRoot();
  const pkgRoot = path.join(repoRoot, "services", "email_analyzer");
  const pythonBin = options.pythonPath || process.env.PYTHON_PATH || "python";
  const timeoutMs = options.timeoutMs ?? 600_000;

  const args = ["-m", "email_analyzer", "--project", projectName];
  if (options.period) {
    args.push("--period", String(options.period));
  }
  if (options.days != null) {
    args.push("--days", String(options.days));
  }
  if (options.noLlm) {
    args.push("--no-llm");
  }
  if (options.assistantProvider) {
    args.push("--assistant-provider", options.assistantProvider);
  }

  return new Promise((resolve, reject) => {
    const env = {
      ...process.env,
      PYTHONPATH: [pkgRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
    };

    const child = spawn(pythonBin, args, {
      cwd: repoRoot,
      env,
      windowsHide: true,
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Timeout ${timeoutMs}ms`));
    }, timeoutMs);

    child.stdout?.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr?.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });

    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr || `python exited with code ${code}`));
        return;
      }
      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch (e) {
        reject(new Error(`Invalid JSON: ${e}\n${stdout.slice(0, 2000)}`));
      }
    });
  });
}
