import functools
from datetime import datetime

import tensorflow as tf

def add_summary(name, summary_type):
    """ decorator to add the result of collection """
    def make_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            summary_type(name, result)

            return result
        return wrapper
    return make_decorator

def print_with_time(string):
    """ print with time """
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] {string}')
