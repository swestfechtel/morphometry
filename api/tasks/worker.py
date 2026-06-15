"""RQ worker entrypoint.

Run with: ``python -m api.tasks.worker`` (or the standard ``rq worker -u $REDIS_URL gpu``).
The worker is the only process that loads full volumes and runs docker, so it
needs docker-socket access and the morphometry environment. Keep exactly one
worker on the GPU queue so model jobs are serialized.
"""
import logging

from redis import Redis
from rq import Queue, Worker

from api.logging_config import configure_logging
from api.runtime import get_engine
from api.settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_dir, settings.log_level, name="api")
    get_engine()  # ensure tables exist
    connection = Redis.from_url(settings.redis_url)
    queue = Queue(settings.gpu_queue_name, connection=connection)
    logging.getLogger("api").info("Starting RQ worker on queue '%s'", settings.gpu_queue_name)
    Worker([queue], connection=connection).work()


if __name__ == "__main__":
    main()
