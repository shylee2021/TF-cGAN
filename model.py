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

            h1_1 = tf.layers.dense(noises, 200, activation=leaky_relu, use_bias=False,
                                   kernel_initializer=tf.initializers.truncated_normal(stddev=0.02) ,name='dense1_noise')

            h1_2 = tf.layers.dense(onehot_labels, 1000, activation=leaky_relu, use_bias=False,
                                   kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='dense1_label')

            h1_c = tf.concat([h1_1, h1_2], axis=1, name='dense1_combined')
            d1_c = tf.layers.batch_normalization(h1_c, momentum=0.8, training=is_training, name='bn1')

            h2 = tf.layers.dense(d1_c, 1200, activation=leaky_relu, use_bias=False,
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='dense2')
            d2 = tf.layers.batch_normalization(h2, momentum=0.8, training=is_training, name='bn2')

            h3 = tf.layers.dense(d2, 1200, activation=leaky_relu, use_bias=False,
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='dense3')
            d3 = tf.layers.batch_normalization(h3, momentum=0.8, training=is_training, name='bn3')

            h4 = tf.layers.dense(d3, 1000, activation=leaky_relu, use_bias=False,
                                 kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='dense4')
            d4 = tf.layers.batch_normalization(h4, momentum=0.8, training=is_training, name='bn4')

            h5 = tf.layers.dense(d4, 784, use_bias=False, kernel_initializer=tf.initializers.truncated_normal(stddev=0.02), name='dense5')
            sig = tf.nn.tanh(h5, name='tanh')

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
            d1 = tf.layers.batch_normalization(h1, momentum=0.8, training=is_training, name='bn1')
            h2 = tf.layers.dense(d1, 256, activation=leaky_relu, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense2')
            d2 = tf.layers.batch_normalization(h2, momentum=0.8, training=is_training, name='bn2')
            h3 = tf.layers.dense(d2, 1, use_bias=True,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense3')
            outputs = tf.nn.sigmoid(h3, name='probability')

            return outputs

    def generate_images(self, labels):
        '''
        generate images with given noises

        :param labels:
        :return:
        '''
        dim = tf.convert_to_tensor([tf.shape(labels)[0], self.noise_len], dtype=tf.int32)
        noises = tf.random.truncated_normal(dim)
        generated = self.generator(noises, labels)
        return generated

    @add_summary('generator_loss', tf.summary.scalar)
    def generator_loss(self, real_targets, fake_predictions):
        ''' loss of generator with given predictions of discriminator '''
        with tf.name_scope('generator_loss'):
            loss = tf.losses.log_loss(real_targets, fake_predictions)
        return loss

    @add_summary('discriminator_loss', tf.summary.scalar)
    def discriminator_loss(self, real_targets, fake_targets, real_predictions, fake_predictions, flip=False):
        ''' loss of discriminator with given predictions '''
        with tf.name_scope('discriminator_loss'):

            loss = tf.cond(
                flip,
                lambda: 0.5 * tf.losses.log_loss(real_targets, real_predictions)
                        + 0.5 * tf.losses.log_loss(fake_targets, fake_predictions),
                lambda: 0.5 * tf.losses.log_loss(fake_targets, real_predictions)
                        + 0.5 * tf.losses.log_loss(real_targets, fake_predictions)
            )

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

    def train(self, sess, dataset, base_lr=3e-8, epochs=300, log_dir='logs/', save_period=None, save_dir='ckpt/'):
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
        noises = tf.random.truncated_normal(dim)

        # outputs of networks
        generated_image = self.generator(noises, labels, is_training=True)
        fake_pred = self.discriminator(generated_image, labels, is_training=True)
        real_pred = self.discriminator(data, labels, is_training=True)

        soft_fake_targets = tf.zeros_like(fake_pred)
        soft_real_targets = tf.ones_like(real_pred) * 0.9

        hard_real_targets = tf.ones_like(real_pred)

        # losses
        gen_loss = self.generator_loss(hard_real_targets, fake_pred)

        flip_prob = tf.random.uniform(shape=[])
        disc_loss = self.discriminator_loss(soft_real_targets, soft_fake_targets, real_pred, fake_pred, flip=(flip_prob > 0.95))

        # learning rate decay
        global_step = tf.Variable(0, trainable=False, name='global_step')
        # lr = tf.maximum(tf.train.exponential_decay(base_lr, global_step, decay_steps=1, decay_rate=(1/1.00004)), 0.000003)
        lr = base_lr

        # operations
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            disc_train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.7).minimize(disc_loss, var_list=self.discriminator_params)
            with tf.control_dependencies([disc_train_op]):
                gen_train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(gen_loss, var_list=self.generator_params, global_step=global_step)

        var_init_op = tf.global_variables_initializer()
        data_iter_init_op = data_iterator.initializer

        # summaries
        tf.summary.image('real_images', tf.expand_dims(data, -1), max_outputs=16)
        # tf.summary.scalar('learning rate', lr)
        tf.summary.image('generated_images', tf.expand_dims(generated_image, -1), max_outputs=16)
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
                    _, disc_loss_value, gen_loss_value, summary_str, flip_val = sess.run([gen_train_op, disc_loss, gen_loss, summaries, flip_prob])

                    flipped = (flip_val > 0.95)
                    step_value = tf.train.global_step(sess, global_step)
                    writer.add_summary(summary_str, global_step=step_value)
                    print_with_time(
                        f'(epoch {epoch}, step {step_value:03}, lr={lr:.3e}) loss of generator: {gen_loss_value:2.6f}, loss of discriminator: {disc_loss_value:2.6f}, flipped: {flipped}')
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