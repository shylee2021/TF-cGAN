import functools

import tensorflow as tf

def reusable_graph(func):
    """
    decorator to wrap tf.make_template() function

    :param function:
    :return:
    """
    return tf.make_template(func.__name__, func, create_scope_now_=True)

class ReusableGraph:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.wrapper(*args, **kwargs)

    def __get__(self, instance, owner):
        """
        getter to decorate the method

        if any kind of method is decorated by this class, getter will be called

        :param instance:
        :param owner:
        :return:
        """

        decorator = self.__call__
        decorator = functools.partial(decorator, instance)
        decorator = functools.wraps(self.func)(decorator)
        return decorator

    def method_decorator(self, instance, *):
        pass


# rg = ReusableGraph(generator)
# rg(x, y, is_training=False)