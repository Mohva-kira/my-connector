import type { SyncEventMap, SyncEventName } from "../types";

type Listener<K extends SyncEventName> = (payload: SyncEventMap[K]) => void;

/**
 * Typed pub/sub singleton decoupling the Zustand store from imperative,
 * one-off animations (floating text, particle bursts, robot reactions).
 * Render state changes flow through normal React re-renders; the EventBus
 * is only for discrete moments that must fire exactly once per occurrence.
 */
class SyncEventBus {
  private readonly listeners: {
    [K in SyncEventName]?: Set<Listener<K>>;
  } = {};

  /** Subscribes to an event. Returns an unsubscribe function. */
  on<K extends SyncEventName>(event: K, listener: Listener<K>): () => void {
    const set = (this.listeners[event] ??= new Set<Listener<K>>() as never);
    (set as Set<Listener<K>>).add(listener);
    return () => {
      (set as Set<Listener<K>>).delete(listener);
    };
  }

  /** Emits an event to every current subscriber, in subscription order. */
  emit<K extends SyncEventName>(event: K, payload: SyncEventMap[K]): void {
    const set = this.listeners[event] as Set<Listener<K>> | undefined;
    if (!set) return;
    for (const listener of set) {
      listener(payload);
    }
  }

  /** Removes every listener for every event. Intended for test teardown. */
  clear(): void {
    for (const key of Object.keys(this.listeners) as SyncEventName[]) {
      delete this.listeners[key];
    }
  }
}

export const syncEventBus = new SyncEventBus();
