'''  '''
import inspect
import os
import shutil

import tensorflow as tf

from utils import add_summary
from utils import print_with_time
from utils import ReusableGraph


class CGAN:
    def __init__(self, name):
        self.name = name
        self.noise_len = 100

        self._build_graph_for_export()

    def generator(self, noises, labels, is_training=False):
        '''

        :param noises:
        :param labels:
        :param is_training:
        :return:
        '''
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            onehot_labels = tf.one_hot(labels, depth=10, dtype=tf.float32)

            h1_1 = tf.layers.dense(noises, 200, activation=tf.nn.relu, use_bias=True, name='dense1_noise')
            h1_2 = tf.layers.dense(onehot_labels, 1000, activation=tf.nn.relu, use_bias=True, name='dense1_label')
            h1_combined = tf.concat([h1_1, h1_2], axis=1, name='dense1_combined')

            h2 = tf.layers.dense(h1_combined, 1200, activation=tf.nn.relu, use_bias=True, name='dense2')

            h3 = tf.layers.dense(h2, 784, name='dense3')
            sig = tf.nn.sigmoid(h3, name='sigmoid')

            return tf.reshape(sig, [-1, 28, 28], name='generated')

    def discriminator(self, inputs, labels, is_training=False):
        '''

        :param inputs:
        :param labels:
        :param is_training:
        :return:
        '''
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            onehot_labels = tf.one_hot(labels, depth=10, dtype=tf.float32)
            flatten_inputs = tf.reshape(inputs, [-1, 28*28])

            conditioned_inputs = tf.concat([flatten_inputs, onehot_labels], axis=1)

            h1 = tf.layers.dense(conditioned_inputs, 256, activation=leaky_relu, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense1')
            #d1 = tf.layers.dropout(h1, rate=0.5, training=is_training, name='dropout1')
            #h2 = tf.layers.dense(h1, 256, activation=leaky_relu, use_bias=True,
            #                     kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
            #d2 = tf.layers.dropout(h2, rate=0.5, training=is_training, name='dropout2')
            h3 = tf.layers.dense(h1, 1, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
            outputs = tf.nn.sigmoid(h3, name='probability')

            return outputs

    def generate_images(self, labels):
        '''
        generate images with given noises

        :param labels:
        :return:
        '''
        dim = tf.convert_to_tensor([tf.shape(labels)[0], self.noise_len], dtype=tf.int32)
        noises = tf.random.uniform(dim)
        generated = self.generator(noises, labels)
        return generated

    @add_summary('generator_loss', tf.summary.scalar)
    def generator_loss(self, real_targets, fake_predictions):
        ''' loss of generator with given predictions of discriminator '''
        with tf.name_scope('generator_loss'):
            loss = tf.losses.log_loss(real_targets, fake_predictions)
        return loss

    @add_summary('discriminator_loss', tf.summary.scalar)
    def discriminator_loss(self, real_targets, fake_targets, real_predictions, fake_predictions):
        ''' loss of discriminator with given predictions '''
        with tf.name_scope('discriminator_loss'):
            loss = (0.5 * tf.losses.log_loss(real_targets, real_predictions)
                    + 0.5 * tf.losses.log_loss(fake_targets, fake_predictions))

        return loss

    @property
    def generator_params(self):
        ''' parameters of generator'''
        return tf.trainable_variables(scope='generator')

    @property
    def discriminator_params(self):
        ''' parameters of discriminator '''
        return tf.trainable_variables(scope='discriminator')

    def _build_graph_for_export(self):
        self.input_label = tf.placeholder(tf.int32, (None,))
        self.generated = self.generate_images(self.input_label)

    def train(self, sess, dataset, base_lr=3e-8, epochs=300, optimizer=tf.train.AdamOptimizer,
              log_dir='logs/', save_period=None, save_dir='ckpt/'):
        '''
        train cGAN model with given dataset

        :param sess:
        :param real_dataset:
        :param label_set:
        :param lr:
        :param epochs:
        :param optimizer:
        :param log_dir:
        :return:
        '''
        # set iterator
        data_iterator = dataset.make_initializable_iterator()

        # data to feed
        batch = data_iterator.get_next()
        data = batch['data']
        labels = batch['labels']
        dim = tf.convert_to_tensor([tf.shape(data)[0], self.noise_len], dtype=tf.int32)
        noises = tf.random.uniform(dim)

        # outputs of networks
        generated_image = self.generator(noises, labels, is_training=True)
        fake_pred = self.discriminator(generated_image, labels, is_training=True)
        real_pred = self.discriminator(data, labels, is_training=True)

        real_targets = tf.ones_like(real_pred)
        fake_targets = tf.zeros_like(fake_pred)

        # losses
        gen_loss = self.generator_loss(real_targets, fake_pred)
        disc_loss = self.discriminator_loss(real_targets, fake_targets, real_pred, fake_pred)

        # learning rate decay
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.exponential_decay(base_lr, global_step, decay_steps=1000000, decay_rate=0.90)

        # operations
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            disc_train_op = optimizer(learning_rate=lr, beta1=0.5).minimize(disc_loss, var_list=self.discriminator_params)
            with tf.control_dependencies([disc_train_op]):
                gen_train_op = optimizer(learning_rate=lr, beta1=0.5).minimize(gen_loss, var_list=self.generator_params, global_step=global_step)

        var_init_op = tf.global_variables_initializer()
        data_iter_init_op = data_iterator.initializer

        # summaries
        tf.summary.scalar('learning rate', lr)
        tf.summary.image('generated_images', tf.expand_dims(generated_image, -1))
        summaries = tf.summary.merge_all()

        # set logger
        log_path = os.path.join(log_dir, self.name)
        writer = tf.summary.FileWriter(log_path, sess.graph)

        # set saver
        saver = tf.train.Saver()
        save_path = os.path.join(save_dir, self.name)

        sess.run(var_init_op)
        for epoch in range(epochs):
            sess.run(data_iter_init_op)

            while True:
                try:
                    _, disc_loss_value, gen_loss_value, summary_str, lr_value = sess.run([gen_train_op, disc_loss, gen_loss, summaries, lr])

                    step_value = tf.train.global_step(sess, global_step)
                    writer.add_summary(summary_str, global_step=step_value)
                    print_with_time(
                        f'(epoch {epoch}, step {step_value:03}, lr={lr_value:.6e}) loss of generator: {gen_loss_value:.6f}, loss of discriminator: {disc_loss_value:.6f}')
                except tf.errors.OutOfRangeError:
                    break

            if save_period is not None and (epoch+1) % save_period == 0:
                saver.save(sess, save_path, global_step=global_step)

    def export(self, sess, export_dir):
        '''
        export servable tensorflow model

        :param sess: `tf.Session()`
        :param export_dir: str to directory to export
        '''
        if os.path.exists(export_dir):
            shutil.rmtree(export_dir)

        builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

        input_label_info = tf.saved_model.utils.build_tensor_info(self.input_label)
        generated_info = tf.saved_model.utils.build_tensor_info(self.generated)

        signature_def = tf.saved_model.build_signature_def(
            inputs={'label': input_label_info},
            outputs={'generated': generated_info},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'generated_images': signature_def
            })

        builder.save()

def leaky_relu(x):
    return tf.nn.leaky_relu(x, alpha=0.2)