import tensorflow as tf


def preprocess(data_dict):
    data = data_dict['data']
    data = tf.cast(data, tf.float32)
    data /= 255.0
    data = data * 2.0 - 1.0

    return {'data': data, 'labels': data_dict['labels']}

def get_mnist_dataset(batch_size=128):
    train, test = tf.keras.datasets.mnist.load_data()
    train_x, train_y = train
    test_x, test_y = test

    with tf.name_scope('train_dataset'):
        train_dataset = tf.data.Dataset.from_tensor_slices({'data': train_x, 'labels': train_y})
        train_dataset = train_dataset.map(preprocess) \
                                     .shuffle(100000).batch(batch_size).prefetch(10)

    with tf.name_scope('test_dataset'):
        test_dataset = tf.data.Dataset.from_tensor_slices({'data': test_x, 'labels': test_y})
        test_dataset = test_dataset.map(preprocess)\
                                   .batch(batch_size).prefetch(10)

    return train_dataset, test_dataset
