"""Store de jobs asynchrones adossé à Redis, pour décharger les analyses longues.

Les analyses (`/api/analyze`) sont trop lentes pour tenir dans une requête HTTP
sans dépasser le timeout de la passerelle (Nginx / zrok) → 504. L'exécution
proprement dite est déléguée à un worker ``arq`` (process séparé, voir
``analysis_tasks.py``) ; ce module ne fait que : créer un état de job dans
Redis, mettre ce job en file d'attente auprès du worker, et exposer l'état au
frontend (polling) via ``get_job``.

Remplace l'ancien store en mémoire (``ThreadPoolExecutor`` + dict) qui
contraignait uvicorn à un seul worker (état non partagé entre processus).
Avec Redis, l'API et le(s) worker(s) arq peuvent tourner dans des processus
(et des workers uvicorn) distincts.

Hypothèse de conception : un seul écrivain par ``job_id`` à un instant donné
(le worker arq qui exécute ce job) — pas de verrou distribué nécessaire pour
les mises à jour de progression.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Optional

import redis as redis_lib

from email_analyzer.config import redis_url

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# Durée de rétention d'un job dans Redis (secondes) — gérée nativement par
# EXPIRE, rafraîchie à chaque écriture (voir `_write`). Doit dépasser le plus
# long `job_timeout`/`timeout` arq (run_domain_discovery : 7200s, voir
# analysis_tasks.WorkerSettings) avec une marge, sinon la clé pourrait expirer
# entre deux checkpoints de progression sur un job inhabituellement lent.
_JOB_TTL_SECONDS = 2 * 60 * 60 + 5 * 60

_KEY_PREFIX = "myconnector:job:"

# Throttle de l'auto-sync au login (Fast-Track sur tous les projets actifs
# d'un tenant, voir saas_logic.trigger_login_auto_sync) — au plus 1x/24h/tenant.
_AUTO_SYNC_KEY_PREFIX = "myconnector:auto_sync:"
_AUTO_SYNC_TTL_SECONDS = 24 * 60 * 60

_redis: Optional["redis_lib.Redis"] = None


def _client() -> "redis_lib.Redis":
    global _redis
    if _redis is None:
        _redis = redis_lib.Redis.from_url(redis_url(), decode_responses=True)
    return _redis


def _key(job_id: str) -> str:
    return f"{_KEY_PREFIX}{job_id}"


def _read(job_id: str) -> Optional[Dict[str, Any]]:
    raw = _client().get(_key(job_id))
    return json.loads(raw) if raw is not None else None


def _write(job_id: str, state: Dict[str, Any]) -> None:
    _client().set(_key(job_id), json.dumps(state), ex=_JOB_TTL_SECONDS)


def _update(job_id: str, **fields: Any) -> None:
    state = _read(job_id)
    if state is None:
        return
    state.update(fields)
    _write(job_id, state)


def create_job(tenant_id: Optional[str] = None) -> str:
    """Crée un job en attente dans Redis et retourne son identifiant."""
    job_id = uuid.uuid4().hex
    _write(
        job_id,
        {
            "status": STATUS_PENDING,
            "result": None,
            "error": None,
            "tenant_id": tenant_id,
            "processed": 0,
            "total": 0,
            "partial": None,
        },
    )
    return job_id


def enqueue(job_id: str, task_name: str, *args: Any) -> None:
    """Met ``task_name(ctx, job_id, *args)`` en file d'attente auprès du worker arq.

    ``task_name`` doit correspondre à une fonction enregistrée dans
    ``analysis_tasks.WorkerSettings.functions``. Les arguments doivent être
    sérialisables (types simples : str, int, None) — pas de closures, le
    worker tourne dans un process séparé.
    """
    import asyncio

    from arq import create_pool
    from arq.connections import RedisSettings

    async def _enqueue() -> None:
        pool = await create_pool(RedisSettings.from_dsn(redis_url()))
        try:
            await pool.enqueue_job(task_name, job_id, *args)
        finally:
            await pool.close()

    asyncio.run(_enqueue())


def set_status(
    job_id: str,
    status: str,
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Met à jour le statut (et éventuellement le résultat/l'erreur) d'un job.

    Appelé depuis le worker arq (``analysis_tasks.py``) au fil de l'exécution.
    """
    _update(job_id, status=status, result=result, error=error)


def report_progress(
    job_id: str,
    processed: int,
    total: int,
    partial: Optional[Dict[str, Any]] = None,
) -> None:
    """Met à jour la progression d'un job en cours (callback ``on_batch`` passé
    à l'analyse, exécuté depuis le worker arq)."""
    _update(job_id, processed=processed, total=total, partial=partial)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Retourne l'état du job (status/result/error/tenant_id/processed/total/partial)
    ou ``None`` si absent/expiré."""
    return _read(job_id)


def claim_daily_auto_sync(tenant_id: str) -> bool:
    """Renvoie ``True`` au plus une fois par 24h par tenant (``SET NX EX``,
    claim atomique), ``False`` sinon — évite de relancer un Fast-Track sur
    tous les projets d'un tenant à chaque connexion (voir
    ``saas_logic.trigger_login_auto_sync``)."""
    return bool(
        _client().set(
            f"{_AUTO_SYNC_KEY_PREFIX}{tenant_id}", "1", nx=True, ex=_AUTO_SYNC_TTL_SECONDS
        )
    )
