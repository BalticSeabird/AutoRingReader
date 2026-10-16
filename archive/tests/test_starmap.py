#!/usr/bin/env python3
"""
Quick test to see if starmap is executing and returning results.
"""

import logging
from multiprocessing import get_context
from functools import partial

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def test_worker(idx, name):
    """Simple worker that returns a result"""
    logging.info(f"Worker {idx} processing: {name}")
    return {"id": idx, "name": name, "result": f"processed_{idx}"}

if __name__ == "__main__":
    logging.info("Starting test...")

    tasks = [(i, f"task_{i}") for i in range(5)]

    with get_context('spawn').Pool(2, maxtasksperchild=1) as p:
        logging.info(f"Pool created with 2 processes")

        worker = partial(test_worker)

        logging.info(f"Creating starmap iterator...")
        starmap_iter = p.starmap(worker, tasks)
        logging.info(f"Starmap iterator created")

        result_count = 0
        logging.info(f"Starting iteration...")
        for result in starmap_iter:
            result_count += 1
            logging.info(f"[RESULT {result_count}] Got: {result}")

        logging.info(f"Iteration complete. Received {result_count} results")

    logging.info("Test complete!")
