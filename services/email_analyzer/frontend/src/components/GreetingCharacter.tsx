import { motion, useReducedMotion } from "framer-motion";

type GreetingCharacterProps = {
  size?: number;
  /** false = pose au repos (état "chargement", avant que le chat soit prêt) */
  active?: boolean;
  className?: string;
};

// Illustration codée en SVG (personne assise à un bureau, travaillant sur un
// ordinateur portable) : remplace l'ancien "orbe" (cercle dégradé + halo,
// voir l'historique de index.css). Aucun asset image dans le projet et
// aucun outil de génération d'image/3D disponible ici — le personnage est
// donc entièrement codé à la main, stylisé avec les jetons de couleur du
// thème pour rester cohérent clair/sombre.
export default function GreetingCharacter({
  size = 128,
  active = true,
  className,
}: GreetingCharacterProps) {
  const reduceMotion = useReducedMotion();

  const bobAnimation = reduceMotion ? undefined : { y: [0, -3, 0] };
  const bobTransition = { duration: 3.4, repeat: Infinity, ease: "easeInOut" as const };

  const blinkAnimation = reduceMotion ? undefined : { scaleY: [1, 1, 0.1, 1] };
  const blinkTransition = {
    duration: 0.28,
    repeat: Infinity,
    repeatDelay: 3.6,
    ease: "easeInOut" as const,
    times: [0, 0.55, 0.85, 1],
  };

  const typing = active && !reduceMotion;

  return (
    <motion.svg
      width={size}
      height={size}
      viewBox="0 0 200 200"
      role="img"
      aria-label="Illustration d'une personne travaillant sur un ordinateur"
      className={className}>
      <defs>
        <linearGradient id="greeting-screen-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="rgb(var(--color-ai-primary-rgb))" />
          <stop offset="100%" stopColor="rgb(var(--color-ai-secondary-rgb))" />
        </linearGradient>
      </defs>

      {/* Bureau + chaise : éléments statiques, au sol */}
      <g>
        <path
          d="M55,72 Q49,118 55,156"
          className="stroke-border-default"
          fill="none"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <path
          d="M145,72 Q151,118 145,156"
          className="stroke-border-default"
          fill="none"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <rect
          x="18"
          y="150"
          width="164"
          height="10"
          rx="4"
          className="fill-bg-tertiary stroke-border-default"
          strokeWidth="1.5"
        />
        <line
          x1="30"
          y1="160"
          x2="25"
          y2="190"
          className="stroke-border-default"
          strokeWidth="4"
          strokeLinecap="round"
        />
        <line
          x1="170"
          y1="160"
          x2="175"
          y2="190"
          className="stroke-border-default"
          strokeWidth="4"
          strokeLinecap="round"
        />
      </g>

      {/* Corps : respiration douce (bob), calquée sur le tempo de l'ancien halo */}
      <motion.g animate={bobAnimation} transition={bobTransition}>
        {/* Torse / chemise — contour plus marqué que les jetons "surface"
            habituels : la chemise off-white se fondait dans le fond en
            thème clair avec un simple stroke-border-default. */}
        <path
          d="M70,116 Q70,100 100,100 Q130,100 130,116 L132,152 L68,152 Z"
          fill="#F7F5F1"
          className="stroke-text-secondary"
          strokeWidth="2"
        />

        {/* Bras + mains, tapotement alterné sur le clavier si actif */}
        <motion.g
          animate={typing ? { y: [0, -2, 0] } : undefined}
          transition={{ duration: 0.55, repeat: Infinity, ease: "easeInOut", repeatType: "mirror" }}>
          <path
            d="M74,118 C63,128 60,140 69,148"
            fill="none"
            strokeWidth="10"
            strokeLinecap="round"
            className="stroke-text-secondary/40"
          />
          <path
            d="M74,118 C63,128 60,140 69,148"
            fill="none"
            stroke="#F7F5F1"
            strokeWidth="8"
            strokeLinecap="round"
          />
          <circle cx="69" cy="148" r="6" fill="#C6885C" />
        </motion.g>
        <motion.g
          animate={typing ? { y: [0, -2, 0] } : undefined}
          transition={{
            duration: 0.55,
            repeat: Infinity,
            ease: "easeInOut",
            repeatType: "mirror",
            delay: 0.18,
          }}>
          <path
            d="M126,118 C137,128 140,140 131,148"
            fill="none"
            strokeWidth="10"
            strokeLinecap="round"
            className="stroke-text-secondary/40"
          />
          <path
            d="M126,118 C137,128 140,140 131,148"
            fill="none"
            stroke="#F7F5F1"
            strokeWidth="8"
            strokeLinecap="round"
          />
          <circle cx="131" cy="148" r="6" fill="#C6885C" />
        </motion.g>

        {/* Tête */}
        <g transform="translate(100,78)">
          {/* Cheveux afro, derrière le visage */}
          <g fill="#20140D">
            <circle cx="-24" cy="-6" r="14" />
            <circle cx="-15" cy="-22" r="15" />
            <circle cx="4" cy="-27" r="16" />
            <circle cx="21" cy="-21" r="15" />
            <circle cx="27" cy="-4" r="14" />
            <circle cx="0" cy="-16" r="19" />
          </g>

          {/* Visage */}
          <circle cx="0" cy="2" r="22" fill="#C6885C" />

          {/* Lunettes rondes */}
          <g className="stroke-text-primary" fill="none" strokeWidth="2.2">
            <circle cx="-10" cy="1" r="9" />
            <circle cx="10" cy="1" r="9" />
            <line x1="-1" y1="1" x2="1" y2="1" />
            <line x1="-19" y1="0" x2="-24" y2="-3" />
            <line x1="19" y1="0" x2="24" y2="-3" />
          </g>

          {/* Yeux, clignement périodique */}
          <motion.g
            className="fill-text-primary"
            style={{ transformOrigin: "center" }}
            animate={blinkAnimation}
            transition={blinkTransition}>
            <ellipse cx="-10" cy="1" rx="2.2" ry="2.6" />
            <ellipse cx="10" cy="1" rx="2.2" ry="2.6" />
          </motion.g>

          {/* Sourire */}
          <path
            d="M-6,13 Q0,17 6,13"
            className="stroke-text-primary"
            fill="none"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </g>
      </motion.g>

      {/* Ordinateur portable, posé sur le bureau (statique, devant le corps) */}
      <g>
        <rect
          x="80"
          y="141"
          width="40"
          height="6"
          rx="2"
          className="fill-bg-tertiary stroke-border-default"
          strokeWidth="1"
        />
        <rect x="82" y="112" width="36" height="29" rx="3" fill="url(#greeting-screen-gradient)" />
        {active ? (
          <motion.rect
            x="82"
            y="112"
            width="36"
            height="29"
            rx="3"
            fill="url(#greeting-screen-gradient)"
            animate={reduceMotion ? undefined : { opacity: [0.85, 1, 0.85] }}
            transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
          />
        ) : null}
      </g>
    </motion.svg>
  );
}
