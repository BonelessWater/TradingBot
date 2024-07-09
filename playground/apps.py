from django.apps import AppConfig
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

logger = logging.getLogger(__name__)

class PlaygroundConfig(AppConfig):
    name = 'playground'
    verbose_name = "Playground Application"

    def ready(self):
        self.start_scheduler()

    def start_scheduler(self):
        from .tasks import save_data, save_all_sp500_metrics, daily_cov  # Import your task functions here

        scheduler = BackgroundScheduler()
        
        scheduler.add_job(
            save_data,
            trigger=CronTrigger(hour=0, minute=0),  # Run daily at 12:00 AM
            id='daily_save_data',
            replace_existing=True,
            max_instances=1
        )
        
        scheduler.add_job(
            save_all_sp500_metrics,
            trigger=CronTrigger(hour=0, minute=30),  # Run daily at 12:00 AM
            id='daily_save_all_sp500_metrics',
            replace_existing=True,
            max_instances=1
        )
        
        scheduler.add_job(
            daily_cov,
            trigger=CronTrigger(hour=0, minute=45),  # Run daily at 12:00 AM
            id='daily_cov',
            replace_existing=True,
            max_instances=1
        )
        
        scheduler.start()
        logger.debug("Schedulers have been started for daily tasks.")
