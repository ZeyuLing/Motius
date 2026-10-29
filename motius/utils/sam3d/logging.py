"""SAM component logging uses the host framework logger."""
import logging

def get_pylogger(name=__name__):
    return logging.getLogger(name)
