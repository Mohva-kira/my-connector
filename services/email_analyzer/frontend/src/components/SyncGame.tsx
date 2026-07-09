import { useSyncGame } from "../hooks/useSyncGame";
import { SyncExperience } from "../sync-experience/components/SyncExperience";

/**
 * Public entry point for the gamified sync/loading experience. Mounts the
 * store -> EventBus bridge once, then renders the visual pipeline. This is
 * the only file other pages should import from this module.
 */
export default function SyncGame(): JSX.Element {
  useSyncGame();

  return <SyncExperience />;
}
