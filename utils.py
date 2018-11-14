import functools
from datetime import datetime

import tensorflow as tf


class ReusableGraph:
    def __init__(self, callable_):
        self.callable_ = callable_

    def __call__(self, *args, **kwargs):
        wrapper = tf.make_template(self.callable_.__name__, self.callable_, create_scope_now_=False)
        wrapper = functools.wraps(self.callable_)(wrapper)
        return wrapper(*args, **kwargs)

    def __get__(self, instance, owner):
        return functools.partial(self, instance)

def reusable_graph(func):
    """
    decorator to wrap tf.make_template() function

    :param function:
    :return:
    """

    wrapper = tf.make_template(func.__name__, func, create_scope_now_=False)
    wrapper = functools.wraps(func)(wrapper)

    return wrapper

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
    """print with time"""
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{now}] {string}')
