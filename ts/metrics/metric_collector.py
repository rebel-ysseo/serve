"""
Single start point for system metrics and process metrics script

"""
import argparse
import logging
import sys

from ts.metrics import system_metrics
from ts.metrics.process_memory_metric import check_process_mem_usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", action="store", help="number of GPUs", type=int)
    parser.add_argument("--vendor", action="store", help="vendor of GPU", type=str)
    arguments = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

    system_metrics.collect_all(sys.modules["ts.metrics.system_metrics"], arguments.gpu, arguments.vendor)

    check_process_mem_usage(sys.stdin)
