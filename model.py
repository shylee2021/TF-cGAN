import tensorflow as tf

from utils import reusable_graph

class CGAN:
    def __init__(self, name):
        self.name = name

    def train(self, sess, real_dataset, label_set, lr=0.0001, epochs=200, optimizer=tf.train.MomentumOptimizer,
              log_dir='./logs/'):
        real_dataset_iterator = real_dataset.make_initializable_iterator()
        label_iterator = label_set.make_initializable_iterator()

        var_init_op = tf.global_variables_initializer()
        iter_init_op = label_iterator.initializer
        gen_train_op = optimizer(learning_rate=lr).minimize(self.generator_loss, var_list=self.generator_params)
        disc_train_op = optimizer(learning_rate=lr).minimize(self.discriminator_loss, var_list=self.discriminator_params)

        labels = label_iterator.get_next()
        noises = tf.random_normal(labels.shape)

        generated_image = self.generator(noises, labels, is_training=True)


    def save_ckpt(self):
        pass

    def export(self):
        pass

    @reusable_graph
    def generator(self, noises, labels, is_training=False):
        conditioned_inputs = tf.concat([noises, labels], axis=1)

        h1 = tf.layers.dense(conditioned_inputs, 512, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense1')
        d1 = tf.layers.dropout(h1, rate=0.5, training=is_training, name='dropout1')
        h2 = tf.layers.dense(d1, 512, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
        d2 = tf.layers.dropout(h2, rate=0.5, training=is_training, name='dropout2')
        h3 = tf.layers.dense(d2, 784, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initiallizer(), name='dense3')
        outputs = tf.nn.sigmoid(h3)

        return outputs

    @reusable_graph
    def discriminator(self, inputs, labels, is_training=False):
        conditioned_inputs = tf.concat([inputs, labels], axis=1)

        h1 = tf.layers.dense(conditioned_inputs, 512, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense1')
        d1 = tf.layers.dropout(h1, rate=0.5, training=is_training, name='dropout1')
        h2 = tf.layers.dense(d1, 256, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
        d2 = tf.layers.dropout(h2, rate=0.5, training=is_training, name='dropout2')
        h3 = tf.layers.dense(d2, 10, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense3')
        outputs = tf.nn.sigmoid(h3)

        return outputs

    @property
    def generator_params(self):
        return tf.trainable_variables(scope='generator')

    @property
    def discriminator_params(self):
        return tf.trainable_variables(scope='discriminator')

    @property
    def generator_loss(self):
        return None

    @property
    def discriminator_loss(self):
        return None

