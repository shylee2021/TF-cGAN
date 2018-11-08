import os

import tensorflow as tf
from tensorflow.errors import OutOfRangeError

from utils import print_with_time
from utils import reusable_graph


class CGAN:
    def __init__(self, name):
        self.name = name

    @reusable_graph
    def generator(self, noises, labels, is_training=False):
        '''

        :param noises:
        :param labels:
        :param is_training:
        :return:
        '''
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
        '''

        :param inputs:
        :param labels:
        :param is_training:
        :return:
        '''
        conditioned_inputs = tf.concat([inputs, labels], axis=1)

        h1 = tf.layers.dense(conditioned_inputs, 512, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense1')
        d1 = tf.layers.dropout(h1, rate=0.5, training=is_training, name='dropout1')
        h2 = tf.layers.dense(d1, 256, activation=tf.nn.relu, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
        d2 = tf.layers.dropout(h2, rate=0.5, training=is_training, name='dropout2')
        h3 = tf.layers.dense(d2, 1, use_bias=True,
                             kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense3')
        outputs = tf.nn.sigmoid(h3)

        return outputs

    def generate_images(self, labels):
        '''
        generate images with given noises

        :param labels:
        :return:
        '''
        noises = tf.random_normal(labels.shape)
        return self.generator(noises, labels)

    def generator_loss(self, real_targets, fake_predictions):
        ''' loss of generator with given predictions of discriminator '''
        return tf.losses.log_loss(real_targets, fake_predictions)

    def discriminator_loss(self, real_targets, fake_targets, real_predictions, fake_predictions):
        ''' loss of discriminator with given predictions '''
        return (0.5 * tf.losses.log_loss(real_targets, real_predictions)
                + 0.5 * tf.losses.log_loss(fake_targets, fake_predictions))

    @property
    def generator_params(self):
        ''' parameters of generator'''
        return tf.trainable_variables(scope='generator')

    @property
    def discriminator_params(self):
        ''' parameters of discriminator '''
        return tf.trainable_variables(scope='discriminator')

    def train(self, sess, real_dataset, label_set, lr=0.0001, epochs=200, optimizer=tf.train.MomentumOptimizer,
              log_dir='./logs/', save_period=None, save_dir='./ckpt'):
        """
        train cGAN model with given dataset

        :param sess:
        :param real_dataset:
        :param label_set:
        :param lr:
        :param epochs:
        :param optimizer:
        :param log_dir:
        :return:
        """
        # set iterator
        real_dataset_iterator = real_dataset.make_initializable_iterator()
        label_iterator = label_set.make_initializable_iterator()

        # data to feed
        labels = label_iterator.get_next()
        real_data = real_dataset_iterator.get_next()
        noises = tf.random_normal(labels.shape)

        # outputs of networks
        generated_image = self.generator(noises, labels, is_training=True)
        fake_pred = self.discriminator(generated_image, labels, is_training=True)
        real_pred = self.discriminator(real_data, labels, is_training=True)

        real_targets = tf.ones_like(real_pred)
        fake_targets = tf.zeros_like(fake_pred)

        # losses
        gen_loss = self.generator_loss(real_targets, fake_pred)
        disc_loss = self.discriminator_loss(real_targets, fake_targets, real_pred, fake_pred)

        # operations
        var_init_op = tf.global_variables_initializer()
        label_iter_init_op = label_iterator.initializer
        data_iter_init_op = real_dataset_iterator.initializer
        gen_train_op = optimizer(learning_rate=lr).minimize(gen_loss, var_list=self.generator_params)
        disc_train_op = optimizer(learning_rate=lr).minimize(disc_loss, var_list=self.discriminator_params)

        # summaries
        # TODO(shylee2021): implement summaries
        summaries = None

        # set logger
        sess.run(var_init_op)
        log_path = os.path.join(log_dir, self.name)
        writer = tf.summary.FileWriter(log_path, sess.graph)

        global_step = 0
        for epoch in range(epochs):
            sess.run(label_iter_init_op)
            sess.run(data_iter_init_op)

            while True:
                try:
                    _, summary_str = sess.run([gen_train_op, summaries])
                    _, summary_str = sess.run([disc_train_op, summaries])
                    loss_disc, loss_gen = sess.run([disc_loss, gen_loss])

                    writer.add_summary(summary_str, global_step=global_step)
                    print_with_time(
                        '(epoch {epoch}, step {step:03}, lr={lr}) loss of generator: {loss_gen}, loss of discriminator: {loss_disc}' \
                            .format(epoch=epoch, step=global_step, loss_gen=loss_gen, loss_disc=loss_disc, lr=lr))
                except OutOfRangeError:
                    break
                else:
                    global_step += 1

            if save_period is not None and (epoch+1) / save_period == 0:
                self.save_ckpt(save_dir)

    def save_ckpt(self, save_dir='./ckpt'):
        pass

    def export(self):
        pass