/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Palette "bureau" claire et sobre (remplace le thème sombre
        // indigo/tech) : blanc cassé/gris très clair pour le fond, bleu
        // profond pour les accents interactifs, vert/orange/rouge réservés
        // aux statuts sémantiques (santé projet/attention/urgence critique).
        // Mêmes noms de jetons qu'avant — aucun composant à retoucher pour
        // ce changement seul (voir context/ui-context.md).
        "bg-primary": "#FAFAF8",
        "bg-secondary": "#F4F4F1",
        "bg-tertiary": "#ECECE7",
        surface: "#FFFFFF",
        "text-primary": "#1E2530",
        "text-secondary": "#59636F",
        "text-muted": "#8A94A0",
        "border-default": "#E2E1DC",
        "border-subtle": "#ECEBE6",
        success: "#16A34A",
        warning: "#D97706",
        danger: "#DC2626",
        info: "#2563EB",
        "ai-primary": "#1E3A8A",
        "ai-secondary": "#3B5BDB",
        "ai-glow": "rgba(30, 58, 138, 0.08)",
        "ai-border": "#1E3A8A",
        "priority-high": "#DC2626",
        "priority-medium": "#D97706",
        "priority-low": "#16A34A",
        "accent-primary": "#1D4ED8",
        "accent-hover": "#1E40AF",
        "accent-soft": "rgba(29, 78, 216, 0.08)",
        "fasttrack-active": "#2563EB",
        "fasttrack-bg": "rgba(37, 99, 235, 0.08)",
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
