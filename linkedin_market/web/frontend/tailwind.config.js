/** @type {import('tailwindcss').Config} */

// Mêmes jetons que services/email_analyzer/frontend (copie : l'outil reste autonome).
function withOpacity(variableName, defaultOpacity = 1) {
  return ({ opacityValue }) => `rgba(var(${variableName}), ${opacityValue ?? defaultOpacity})`;
}

export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        "bg-primary": withOpacity("--color-bg-primary-rgb"),
        "bg-secondary": withOpacity("--color-bg-secondary-rgb"),
        "bg-tertiary": withOpacity("--color-bg-tertiary-rgb"),
        surface: withOpacity("--color-surface-rgb"),
        "text-primary": withOpacity("--color-text-primary-rgb"),
        "text-secondary": withOpacity("--color-text-secondary-rgb"),
        "text-muted": withOpacity("--color-text-muted-rgb"),
        "border-default": withOpacity("--color-border-default-rgb"),
        "border-subtle": withOpacity("--color-border-subtle-rgb"),
        "border-strong": withOpacity("--color-border-strong-rgb"),
        success: withOpacity("--color-success-rgb"),
        warning: withOpacity("--color-warning-rgb"),
        danger: withOpacity("--color-danger-rgb"),
        info: withOpacity("--color-info-rgb"),
        "success-strong": withOpacity("--color-success-strong-rgb"),
        "warning-strong": withOpacity("--color-warning-strong-rgb"),
        "danger-strong": withOpacity("--color-danger-strong-rgb"),
        "info-strong": withOpacity("--color-info-strong-rgb"),
        "accent-strong": withOpacity("--color-accent-strong-rgb"),
        "on-accent": withOpacity("--color-on-accent-rgb"),
        overlay: withOpacity("--color-overlay-rgb"),
        "accent-primary": withOpacity("--color-accent-primary-rgb"),
        "accent-hover": withOpacity("--color-accent-hover-rgb"),
        "accent-soft": "rgba(var(--color-accent-soft-rgb), 0.08)",
      },
      fontFamily: { sans: ["Inter Variable", "Inter", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "monospace"] },
      borderRadius: { sm: "6px", md: "10px", lg: "14px", xl: "18px", full: "999px" },
      boxShadow: { card: "var(--shadow-card)", lift: "var(--shadow-lift)" },
    },
  },
  plugins: [],
};
