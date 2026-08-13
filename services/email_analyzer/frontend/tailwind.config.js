/** @type {import('tailwindcss').Config} */

// Jetons de couleur résolus via des variables CSS `--color-*-rgb` (triplet
// "R, G, B") définies dans src/index.css sous `:root` (clair) et
// `[data-theme="dark"]` (sombre). Ce helper reproduit le comportement
// natif de Tailwind pour les modificateurs d'opacité (`bg-danger/10`,
// `bg-ai-glow/40`, etc.) — impossible à obtenir avec un simple
// `var(--x)` puisque Tailwind ne peut pas réécrire l'alpha d'une couleur
// déjà résolue par le navigateur. `defaultOpacity` restaure la valeur
// utilisée aujourd'hui quand le jeton est utilisé sans modificateur
// (ex. `ai-glow`/`accent-soft`/`fasttrack-bg` sont des halos toujours
// semi-transparents par défaut).
function withOpacity(variableName, defaultOpacity = 1) {
  return ({ opacityValue }) => `rgba(var(${variableName}), ${opacityValue ?? defaultOpacity})`;
}

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Mêmes noms de jetons qu'avant — aucun composant à retoucher pour
        // ce changement seul (voir context/ui-context.md).
        "bg-primary": withOpacity("--color-bg-primary-rgb"),
        "bg-secondary": withOpacity("--color-bg-secondary-rgb"),
        "bg-tertiary": withOpacity("--color-bg-tertiary-rgb"),
        surface: withOpacity("--color-surface-rgb"),
        "text-primary": withOpacity("--color-text-primary-rgb"),
        "text-secondary": withOpacity("--color-text-secondary-rgb"),
        "text-muted": withOpacity("--color-text-muted-rgb"),
        "border-default": withOpacity("--color-border-default-rgb"),
        "border-subtle": withOpacity("--color-border-subtle-rgb"),
        success: withOpacity("--color-success-rgb"),
        warning: withOpacity("--color-warning-rgb"),
        danger: withOpacity("--color-danger-rgb"),
        info: withOpacity("--color-info-rgb"),
        "ai-primary": withOpacity("--color-ai-primary-rgb"),
        "ai-secondary": withOpacity("--color-ai-secondary-rgb"),
        // Halos toujours semi-transparents par défaut : jetons statiques
        // (pas withOpacity) car Tailwind force sa propre variable
        // d'opacité (--tw-bg-opacity, repart de 1) sur toute couleur
        // exprimée en fonction — impossible d'y faire porter une opacité
        // par défaut ≠ 1. Composés dans index.css à partir du triplet RGB
        // ci-dessus + une opacité fixe.
        "ai-glow": "rgba(var(--color-ai-glow-rgb), 0.08)",
        "ai-border": withOpacity("--color-ai-border-rgb"),
        "priority-high": withOpacity("--color-priority-high-rgb"),
        "priority-medium": withOpacity("--color-priority-medium-rgb"),
        "priority-low": withOpacity("--color-priority-low-rgb"),
        "accent-primary": withOpacity("--color-accent-primary-rgb"),
        "accent-hover": withOpacity("--color-accent-hover-rgb"),
        "accent-soft": "rgba(var(--color-accent-soft-rgb), 0.08)",
        "fasttrack-active": withOpacity("--color-fasttrack-active-rgb"),
        "fasttrack-bg": "rgba(var(--color-fasttrack-bg-rgb), 0.08)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        sm: "6px",
        md: "10px",
        lg: "14px",
        xl: "18px",
        full: "999px",
      },
    },
  },
  plugins: [],
};
