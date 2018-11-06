import tensorflow as tf

class CGAN:
    def __init__(self, name, inputs, labels):
        self.name = name

    def train(self, sess, dataset, label_set, lr=0.0001, epochs=200, optimizer=tf.train.MomentumOptimizer,
              log_dir='./logs/'):
        pass

    def save_ckpt(self):
        pass

    def export(self):
        pass

    def generator(self, inputs, labels, reuse=False, is_training=False):
        with tf.variable_scope('generator') as v_scope:
            conditioned_inputs = tf.concat([inputs, labels], axis=1)

            h1 = tf.layers.dense(conditioned_inputs, 512, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initiallizer, name='dense1', reuse=reuse)
            d1 = tf.layers.dropout(h1, rate=0.5, training=is_training, name='dropout1')
            h2 = tf.layers.dense(d1, 512, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initiallizer, name='dense2', reuse=reuse)
            d2 = tf.layers.dropout(h2, rate=0.5, training=is_training, name='dropout2')
            h3 = tf.layers.dense(d2, 784, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initiallizer, name='dense3', reuse=reuse)
            outputs = tf.nn.sigmoid(h3)

            return outputs

    def discriminator(self, inputs, labels, reuse=False, is_training=False):
        with tf.variable_scope('discriminator') as v_scope:
            conditioned_inputs = tf.concat([inputs, labels], axis=1)

            h1 = tf.layers.dense(conditioned_inputs, 512, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initiallizer, name='dense1', reuse=reuse)
            d1 = tf.layers.dropout(h1, rate=0.5, training=is_training, name='dropout1')
            h2 = tf.layers.dense(d1, 256, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initiallizer, name='dense2', reuse=reuse)
            d2 = tf.layers.dropout(h2, rate=0.5, training=is_training, name='dropout2')
            h3 = tf.layers.dense(d2, 10, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initiallizer, name='dense3', reuse=reuse)
            outputs = tf.nn.sigmoid(h3)

            return outputs
