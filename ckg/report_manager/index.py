import os
import subprocess

from ckg import ckg_utils
from ckg.graphdb_builder import builder_utils
from ckg.report_manager.app import start_app

try:
    ckg_config = ckg_utils.read_ckg_config()
    log_config = ckg_config['report_manager_log']
    logger = builder_utils.setup_logging(log_config, key="index")
    config = builder_utils.setup_config('builder')
    separator = config["separator"]
except Exception as err:
    logger.error("Reading configuration > {}.".format(err))


def main():
    logger.info("Starting CKG App")
    celery_working_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(celery_working_dir)
    queues = [('creation', 1, 'INFO'), ('compute', 3, 'INFO'), ('update', 1, 'INFO')]
    for queue, processes, log_level in queues:
        celery_cmdline = 'celery -A ckg.report_manager.worker worker --loglevel={} --concurrency={} -E -Q {}'.format(
            log_level, processes, queue).split(" ")
        logger.info("Ready to call {} ".format(celery_cmdline))
        subprocess.Popen(celery_cmdline)
        logger.info("Done calling {} ".format(celery_cmdline))
    start_app()


if __name__ == '__main__':
    main()
