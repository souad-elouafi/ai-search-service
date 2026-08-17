from apscheduler.schedulers.background import BackgroundScheduler
from app.services.backup_sync_service import run_backup_sync
from app.config import (
    BACKUP_SYNC_INTERVAL_MINUTES,
    CHEDMED_API_BASE_URL,
    ENABLE_BACKUP_SCHEDULER,
)

scheduler = BackgroundScheduler()


def _safe_backup_sync():
    """Execute la synchro de secours en avalant les erreurs
    (pour ne jamais faire planter le scheduler en arriere-plan)."""
    if not CHEDMED_API_BASE_URL:
        print("Synchronisation de secours ignoree : CHEDMED_API_BASE_URL non configure.")
        return
    try:
        run_backup_sync()
    except Exception as e:
        print(f"Erreur lors de la synchronisation de secours automatique: {e}")


def start_scheduler():
    """Start one in-process job when explicitly enabled for this process."""
    if not ENABLE_BACKUP_SCHEDULER:
        return False
    if scheduler.running:
        return False
    scheduler.add_job(
        _safe_backup_sync,
        "interval",
        minutes=BACKUP_SYNC_INTERVAL_MINUTES,
        id="backup_sync_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    print(
        "Scheduler demarre : synchronisation de secours toutes les "
        f"{BACKUP_SYNC_INTERVAL_MINUTES} minutes."
    )
    return True


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
