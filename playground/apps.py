from django.apps import AppConfig
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

#logger = logging.getLogger(__name__)

class PlaygroundConfig(AppConfig):
    name = 'playground'
    verbose_name = "Playground Application"

    def ready(self):
        from .tasks import save_data  # Import your task function here
        self.start_scheduler(save_data)

    def start_scheduler(self, task_function):
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            task_function,
            trigger=CronTrigger(hour=0, minute=0),  # Run daily at 12:00 M
            id='daily_data_save',
            replace_existing=True,
            max_instances=1
        )
        scheduler.add_job(
            task_function,
            trigger='date',  # Run once at startup
            id='startup_data_save'
        )
        scheduler.start()
        #logger.debug("Schedulers have been started for data saving.")
