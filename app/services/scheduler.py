from apscheduler.schedulers.background import BackgroundScheduler
from app.services.backup_sync_service import run_backup_sync
from app.config import CHEDMED_API_BASE_URL

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
    """Demarre la tache periodique (toutes les 15 minutes)."""
    scheduler.add_job(_safe_backup_sync, "interval", minutes=15, id="backup_sync_job")
    scheduler.start()
    print("Scheduler demarre : synchronisation de secours toutes les 15 minutes.")


def stop_scheduler():
    scheduler.shutdown(wait=False) 