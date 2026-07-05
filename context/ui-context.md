# UI Color Token System

This design system is built for an AI-powered email intelligence assistant focused on clarity, reduced cognitive load, and decision-making. The aesthetic is minimal, sober, ergonomic, and highly readable.

---

## 🎨 Color Palette (Semantic Tokens)

| Token Name | Hex Value | Usage |
|------------|----------|------|
| `color-bg-primary` | #0B0F17 | Main app background (dark, calm UI base) |
| `color-bg-secondary` | #111827 | Panels, cards, elevated surfaces |
| `color-bg-tertiary` | #1F2937 | Hover states, subtle containers |
| `color-surface` | #111A2E | Main content surfaces (briefing cards) |

| `color-text-primary` | #E5E7EB | Primary text (high readability) |
| `color-text-secondary` | #9CA3AF | Secondary text (metadata, timestamps) |
| `color-text-muted` | #6B7280 | Disabled / low importance text |

| `color-border-default` | #243244 | Default borders |
| `color-border-subtle` | #1B2433 | Soft separators |

---

## 🎯 Semantic Status Colors (AI + Notifications)

| Token Name | Hex Value | Usage |
|------------|----------|------|
| `color-success` | #22C55E | Completed actions, resolved emails |
| `color-warning` | #F59E0B | Important emails, medium priority |
| `color-danger` | #EF4444 | Urgent emails, critical alerts |
| `color-info` | #3B82F6 | Informational emails, system updates |

---

## 🤖 AI Intelligence Colors (Core Differentiation Layer)

| Token Name | Hex Value | Usage |
|------------|----------|------|
| `color-ai-primary` | #6366F1 | AI-generated insights, highlights |
| `color-ai-secondary` | #8B5CF6 | Secondary AI annotations |
| `color-ai-glow` | rgba(99, 102, 241, 0.15) | Subtle AI emphasis background |
| `color-ai-border` | #4F46E5 | AI-highlighted components |

---

## 🔔 Notification & Priority System

| Token Name | Hex Value | Usage |
|------------|----------|------|
| `color-priority-high` | #EF4444 | Urgent email notifications |
| `color-priority-medium` | #F97316 | Important but not urgent |
| `color-priority-low` | #22C55E | Low priority informational emails |

---

## 🧭 Accent Color System

| Token Name | Hex Value | Usage |
|------------|----------|------|
| `color-accent-primary` | #6366F1 | Main interactive elements (buttons, links) |
| `color-accent-hover` | #4F46E5 | Hover state for primary accent |
| `color-accent-soft` | rgba(99, 102, 241, 0.12) | Background highlight states |

---

## 📝 Typography System

| Token | Value | Usage |
|-------|------|------|
| `font-family-base` | Inter, system-ui, sans-serif | Main UI font |
| `font-family-mono` | JetBrains Mono, monospace | Logs, AI reasoning, metadata |

### Font Sizes

| Token | Size | Usage |
|-------|------|------|
| `text-xs` | 12px | Metadata, timestamps |
| `text-sm` | 14px | Secondary UI text |
| `text-base` | 16px | Default body text |
| `text-lg` | 18px | Section labels |
| `text-xl` | 20px | Card titles |
| `text-2xl` | 24px | Dashboard headings |
| `text-3xl` | 30px | Main briefing title |

---

## 📐 Border Radius Scale

| Token | Value | Usage |
|-------|------|------|
| `radius-sm` | 6px | Inputs, small elements |
| `radius-md` | 10px | Buttons, tags |
| `radius-lg` | 14px | Cards, panels |
| `radius-xl` | 18px | Main containers |
| `radius-full` | 999px | Pills, avatars |

---

## 🧠 UX Principles Embedded in Tokens

- Prioritize **readability over decoration**
- Use AI color only for **insights, not decoration**
- Keep danger/warning colors reserved for **true urgency**
- Maintain strong contrast for accessibility (dark-first design)
- Avoid visual noise in non-actionable content

---
