import tensorflow as tf


def get_mnist_dataset(batch_size=100):
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    test_x, test_y = test

    train_dataset = tf.data.Dataset.from_tensor_slices({'data': train_x, 'labels': train_y})
    train_dataset = train_dataset.shuffle(100000).repeat().batch(batch_size).prefetch(1)

    test_dataset = tf.data.Dataset.from_tensor_slices({'data': test_x, 'labels': test_y})
    test_dataset = test_dataset.shuffle(100000).repeat().batch(batch_size).prefetch(1)

    return train_dataset, test_dataset
