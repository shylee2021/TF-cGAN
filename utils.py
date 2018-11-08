import functools
from datetime import datetime

import tensorflow as tf

def reusable_graph(func):
    """
    decorator to wrap tf.make_template() function

    :param function:
    :return:
    """
    return tf.make_template(func.__name__, func, create_scope_now_=True)

def print_with_time(string):
    """print with time"""
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    print('[{t}] {s}'.format(t=now, s=string))
