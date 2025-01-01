import logging
from logging.handlers import QueueHandler, QueueListener
import multiprocessing as mp
from pathlib import Path
import sys
import time


class Logger:
    def __init__(
        self,
        root_path: Path,
        name: str = "log",
        level: int = logging.INFO,
        log_to_stream: bool = False,
    ):
        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=False)

        timestamp = time.strftime("%Y%m%d%H%M%S")
        file_name = f"{timestamp}_{name}.txt"
        logger_path = root_path.joinpath(file_name)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(logger_path, mode="a")
        file_handler.setFormatter(formatter)

        logger = logging.getLogger()
        logger.setLevel(level)
        queue = mp.Queue()
        handlers = [
            file_handler,
        ]
        if log_to_stream:
            handlers.append(logging.StreamHandler(sys.stdout))
        self.logger_queue_listener = QueueListener(queue, *handlers)
        logger.addHandler(QueueHandler(queue))

    def start(self):
        self.logger_queue_listener.start()

    def stop(self):
        self.logger_queue_listener.stop()
