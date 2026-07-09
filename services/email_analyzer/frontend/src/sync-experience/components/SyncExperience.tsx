import { HUD } from "./HUD";

/**
 * Step 3: renders the real HUD. Still not mounted anywhere in the app
 * (see HomePage.tsx) until visually verified. Replaced incrementally by
 * PipelineTrack, Robot, EmailEnvelope and FloatingText in later steps.
 */
export function SyncExperience(): JSX.Element {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-3 p-8">
      <HUD />
    </div>
  );
}
