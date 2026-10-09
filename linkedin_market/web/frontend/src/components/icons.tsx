// Icônes au trait (24×24, couleur héritée) : évite les emojis et une dépendance de plus.

const PATHS = {
  chart: "M4 19V5M4 19h16M8 15v-4M12 15V8M16 15v-6",
  trend: "M3 17l6-6 4 4 8-8M15 7h6v6",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 12h.01",
  scale: "M12 3v18M5 7h14M5 7l-3 7a3 3 0 0 0 6 0L5 7zM19 7l-3 7a3 3 0 0 0 6 0l-3-7zM8 21h8",
  user: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4 21a8 8 0 0 1 16 0",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5M12 8h.01",
  spark: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z",
  play: "M7 5l12 7-12 7V5z",
  external: "M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  copy: "M9 9h10v10H9zM5 15V5h10",
  check: "M5 12l5 5 9-10",
  plus: "M12 5v14M5 12h14",
  thumbUp: "M7 11v9H4v-9h3zM7 11l4-7a2 2 0 0 1 2 2v4h5a2 2 0 0 1 2 2.3l-1 6A2 2 0 0 1 17 20H7",
  thumbDown: "M17 13V4h3v9h-3zM17 13l-4 7a2 2 0 0 1-2-2v-4H6a2 2 0 0 1-2-2.3l1-6A2 2 0 0 1 7 4h10",
  doc: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  users: "M9 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM2 21a7 7 0 0 1 14 0M17 3.5a4 4 0 0 1 0 7M19 21a7 7 0 0 0-3-5.7",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 7v5l3 2",
  pin: "M12 21s-7-6.2-7-11a7 7 0 0 1 14 0c0 4.8-7 11-7 11zM12 12a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
  refresh: "M20 11a8 8 0 0 0-14.9-3M4 4v4h4M4 13a8 8 0 0 0 14.9 3M20 20v-4h-4",
  arrowRight: "M5 12h14M13 6l6 6-6 6",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, className = "h-4 w-4" }: { name: IconName; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}
