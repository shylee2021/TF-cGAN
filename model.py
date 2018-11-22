import os
from pathlib import Path
import shutil
import tensorflow as tf

from utils import add_summary
from utils import print_with_time


class CGAN:
    """ Conditional GAN model

    Attributes:
        name (str): The name of model.
        noise_len (int): The length of input noise.

    References:
        https://arxiv.org/abs/1411.1784
        \"Conditional Generative Adversarial Nets\", Mirza & Osindero
    """

    def __init__(self, name: str):
        self.name = name
        self.noise_len = 100

        self._build_graph_for_export()

    def generator(self, noises, labels, is_training=False):
        """ Generator model function

        Args:
            noises: `tf.Tensor` of shape `[batch_size, length]`. The noises.
            labels: `tf.Tensor` of shape `[batch_size]`. The labels.
            is_training: Tensorflow boolean scalar tensor or Python boolean.
                Whether the model is in training or inference mode.

        Returns:
            Generated image tensor of shape `[batch_size, height, width]`.

        """
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            weight_init = tf.contrib.layers.xavier_initializer()

            onehot_labels = tf.one_hot(labels, depth=10, dtype=tf.float32)

            combined = tf.concat([noises, onehot_labels], axis=1, name='combined')
            dense1 = tf.layers.dense(combined, 128, activation=tf.nn.relu,
                                     kernel_initializer=weight_init, name='dense1')
            dense2 = tf.layers.dense(dense1, 28 * 28, activation=None,
                                     kernel_initializer=weight_init, name='dense2')
            flat_outputs = tf.nn.tanh(dense2, name='tanh')

            return tf.reshape(flat_outputs, [-1, 28, 28], name='generated')

    def discriminator(self, inputs, labels, is_training=False):
        """ Discriminator model function.

        Args:
            inputs: `tf.Tensor` of shape `[batch_size, height, width]`. The image tensors.
                Each element in tensor should be in range [0, 1].
            labels: `tf.Tensor` of shape `[batch_size]`. The labels.
            is_training: Tensorflow boolean scalar tensor or Python boolean.
                Whether the model is in training or inference mode.

        Returns:
            Probability that the input is from real dataset. Tensor of shape `[batch_size]`.
        """
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            weight_init = tf.contrib.layers.xavier_initializer()

            onehot_labels = tf.one_hot(labels, depth=10, dtype=tf.float32)
            flatten_inputs = tf.reshape(inputs, [-1, 28 * 28])

            conditioned_inputs = tf.concat([flatten_inputs, onehot_labels], axis=1, name='conditioned_inputs')
            dense1 = tf.layers.dense(conditioned_inputs, 128, activation=leaky_relu,
                                     kernel_initializer=weight_init, name='dense1')
            dense2 = tf.layers.dense(dense1, 1, activation=None,
                                     kernel_initializer=weight_init, name='dense2')
            outputs = tf.nn.sigmoid(dense2, name='probability')

            return outputs

    def generate_images(self, labels):
        """ Generates image with given labels.

        Args:
            labels: `tf.Tensor` of shape `[batch_size]`. The labels.

        Returns:
            Generated images. A tensor of shape `[batch_size, height, width]`.
        """
        dim = tf.convert_to_tensor([tf.shape(labels)[0], self.noise_len], dtype=tf.int32)
        noises = tf.random.truncated_normal(dim)
        generated = self.generator(noises, labels)
        return generated

    @add_summary('generator_loss', tf.summary.scalar)
    def generator_loss(self, targets, fake_predictions):
        """ Figures out loss of generator with given predictions of discriminator

        Args:
            targets: Groundtruth target to train generator.
            fake_predictions: Predictions of discriminator for generated images.

        Returns:
            Loss tensor to update generator.
        """
        with tf.name_scope('generator_loss'):
            loss = tf.losses.log_loss(targets, fake_predictions)
        return loss

    @add_summary('discriminator_loss', tf.summary.scalar)
    def discriminator_loss(self, real_targets, fake_targets, real_predictions, fake_predictions):
        """ Figure out loss of discriminator with given predictions.

        Args:
            real_targets: Groundtruth targets of real data to train discriminator.
            fake_targets: Groundtruth targets of generated data to train discriminator.
            real_predictions: Predictions of discriminator for real images.
            fake_predictions: Predictions of discriminator for generated images.

        Returns:
            Loss tensor to update discriminator.
        """
        with tf.name_scope('discriminator_loss'):
            loss = 0.5 * tf.losses.log_loss(real_targets, real_predictions) \
                   + 0.5 * tf.losses.log_loss(fake_targets, fake_predictions)

        return loss

    @property
    def generator_params(self):
        """ Returns parameters of generator. """
        return tf.trainable_variables(scope='generator')

    @property
    def discriminator_params(self):
        """ Returns parameters of generator. """
        return tf.trainable_variables(scope='discriminator')

    def _build_graph_for_export(self):
        """ Build graph for export servable version of model """
        self.input_label = tf.placeholder(tf.int32, (None,))
        self.generated = self.generate_images(self.input_label)

    def get_next_data(self, iterator):
        """ Returns the batch of inputs.

        A batch of real data and labels is obtained from iterator, and the same number of noises are formed.

        Args:
            iterator (tf.data.Iterator): Iterator of dataset which consists of data and label dictionary.

        Returns:
            A tuple of (data, labels, noises)
        """
        batch = iterator.get_next()
        data = batch['data']

        labels = batch['labels']

        dim = tf.convert_to_tensor([tf.shape(data)[0], self.noise_len], dtype=tf.int32)
        noises = tf.random.truncated_normal(dim)

        return data, labels, noises

    def get_train_op(self, gen_loss, disc_loss, lr, global_step):
        """ Returns train op with given loss and hyper-parameter

        This function does not return train op for discriminator and generator separately,
        but returns single op that update_ops, discriminator train op and generator op are chained.

        Args:
            gen_loss: The loss of generator.
            disc_loss: The loss of discriminator.
            lr: The learning rate.
            global_step: The global step. The scalar `tf.Variable`.

        Returns:
            Training operation for model.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            disc_train_op = optimizer.minimize(disc_loss, var_list=self.discriminator_params)
            with tf.control_dependencies([disc_train_op]):
                gen_train_op = optimizer.minimize(gen_loss, var_list=self.generator_params, global_step=global_step)

        return gen_train_op

    def train(self, sess, dataset, base_lr=0.0002, epochs=100, log_dir='logs/', save_period=None, save_dir='ckpt/',
              reset_logs=False, version='1'):
        """ Trains cGAN model with given dataset.

        Args:
            sess (tf.Session): Session to train the model with.
            dataset (tf.data.Dataset): Dataset to train the model with.
            base_lr (float): The starting learning rate.
            epochs (int): The number of epochs to train the model. Defaults to 0.0002
            log_dir (str): Directory to save logs for TensorBoard.
            save_period (int, optional): Period to save checkpoint.
                If `save_period` is `None`, checkpoint is not saved during training session.
            save_dir (str): Directory to save checkpoint.
            reset_logs (bool): Flag whether reset all the logs before training.
            versions (str): The version of model.
        """
        # get data to feed
        data_iterator = dataset.make_initializable_iterator()
        data, labels, noises = self.get_next_data(data_iterator)

        # outputs of networks
        generated_image = self.generator(noises, labels, is_training=True)
        pred_fake = self.discriminator(generated_image, labels, is_training=True)
        pred_real = self.discriminator(data, labels, is_training=True)

        targets_fake = tf.zeros_like(pred_fake, name='targets_fake')
        targets_real = tf.ones_like(pred_real, name='targets_real')

        # losses
        gen_loss = self.generator_loss(targets_real, pred_fake)
        disc_loss = self.discriminator_loss(targets_real, targets_fake, pred_real, pred_fake)

        # learning rate
        lr = base_lr

        # global step
        global_step = tf.Variable(0, trainable=False, name='global_step')
        train_op = self.get_train_op(gen_loss, disc_loss, lr, global_step)

        # initialisation op
        var_init_op = tf.global_variables_initializer()
        data_iter_init_op = data_iterator.initializer

        # summaries
        tf.summary.image('real_images', tf.expand_dims(data, -1), max_outputs=16)
        tf.summary.image('generated_images', tf.expand_dims(generated_image, -1), max_outputs=16)
        summaries = tf.summary.merge_all()

        # set logger
        log_path = Path(log_dir) / self.name / version
        if reset_logs and log_path.exists():
            shutil.rmtree(log_path)
        writer = tf.summary.FileWriter(str(log_path), sess.graph)

        # set saver
        saver = tf.train.Saver()
        save_path = Path(save_dir) / self.name / version
        if reset_logs and save_path.exists():
            shutil.rmtree(save_path)

        # train model
        sess.run(var_init_op)
        for epoch in range(epochs):
            sess.run(data_iter_init_op)

            while True:
                try:
                    _, disc_loss_value, gen_loss_value, summary_str = sess.run(
                        [train_op, disc_loss, gen_loss, summaries])

                    step_value = tf.train.global_step(sess, global_step)
                    writer.add_summary(summary_str, global_step=step_value)
                    print_str = (f'(epoch {epoch}, step {step_value:03}, lr={lr:.3e}), '
                                 f'loss of generator: {gen_loss_value:2.6f}, '
                                 f'loss of discriminator: {disc_loss_value:2.6f}')
                    print_with_time(print_str)
                except tf.errors.OutOfRangeError:
                    break

            if save_period is not None and (epoch + 1) % save_period == 0:
                saver.save(sess, str(save_path), global_step=global_step)

    def export(self, sess, export_dir, version):
        """ export servable tensorflow model

        Args:
            sess: Session to save the model with
            export_dir: str to directory to export
            version: version of model
        """
        export_path = Path(export_dir) / version

        if export_path.exists():
            shutil.rmtree(export_path)

        builder = tf.saved_model.builder.SavedModelBuilder(str(export_path))

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
    """ Leaky ReLU with slope 0.2 for negative x range """
    return tf.nn.leaky_relu(x, alpha=0.2)
