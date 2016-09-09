from collections import namedtuple
import zipfile
import numpy

##
# Tensorflow
from tensorflow.python.framework import dtypes
import tensorflow as tf


class DataSet(object):

    def __init__(self,
                 crimes,
                 labels,
                 dtype=dtypes.float64):

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32, dtypes.float64):
            raise TypeError('Invalid crime dtype %r, expected uint8 or float32 or float64' %
                            dtype)

        assert crimes.shape[0] == labels.shape[0], (
            'crimes.shape: %s labels.shape: %s' % (crimes.shape, labels.shape))
        self._num_examples = crimes.shape[0]

        if dtype == dtypes.float32:
            crimes = crimes.astype(numpy.float32)
        elif dtype == dtypes.float64:
            crimes = crimes.astype(numpy.float64)
            ##
            # QUESTO ANDAVA TOLTO!!!
            #crimes = numpy.multiply(crimes, 1.0 / 255.0)

        self._crimes = crimes
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def __len__(self):
        return len(self._crimes)

    @property
    def crimes(self):
        return self._crimes

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._crimes = self._crimes[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._crimes[start:end], self._labels[start:end]


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_crimes(zip_file, filename):
    print('> Extracting data  -> ', filename)
    with zip_file.open(filename, 'r') as bytestream:
        magic = _read32(bytestream)
        if magic != 3584 + 1:  # Vector of double
            raise ValueError('Invalid magic number %d in crimes file: %s' %
                             (magic, filename))
        num_crimes = _read32(bytestream)
        num_features = _read32(bytestream)
        data_size = 8  # bytes, double
        buf = bytestream.read(num_features * num_crimes * data_size)
        data = numpy.frombuffer(buf, dtype=numpy.float64)
        data = data.reshape(num_crimes, num_features)
        # print(data)
        # input()
        return num_features, data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(zip_file, filename):
    print('> Extracting label -> ', filename)
    with zip_file.open(filename, 'r') as bytestream:
        magic = _read32(bytestream)
        if magic != 3072 + 1:  # Vector of integer
            raise ValueError(
                'Invalid magic number %d in crimes label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        num_classes = _read32(bytestream)
        data_size = 4  # bytes
        buf = bytestream.read(num_items * data_size)
        labels = numpy.frombuffer(buf, dtype=numpy.int32)
        return num_classes, dense_to_one_hot(labels, num_classes)


def extract_data(zip_file, files):
    Datasets = namedtuple('Datasets', ['train', 'test'])

    feature_size_train, train_crimes = extract_crimes(
        zip_file, files['train']['set'])
    class_size_train, train_labels = extract_labels(
        zip_file, files['train']['label'])

    feature_size_test, test_crimes = extract_crimes(
        zip_file, files['test']['set'])
    class_size_test, test_labels = extract_labels(
        zip_file, files['test']['label'])

    if feature_size_train != feature_size_test:
        raise Exception("Feature size of train file is different in test file")
    if class_size_train != class_size_test:
        raise Exception("Class size of train file is different in test file")

    feature_size = feature_size_train
    class_size = class_size_train

    train = DataSet(train_crimes, train_labels, dtype=dtypes.float64)
    test = DataSet(test_crimes, test_labels, dtype=dtypes.float64)

    return feature_size, class_size, Datasets(train=train, test=test)


def classify(filename):

    with zipfile.ZipFile(filename) as cur_zip:
        file_trace = {
            'train': {},
            'test': {}
        }

        for name in cur_zip.namelist():
            if '-train-' in name:
                if '-label' in name:
                    file_trace['train']['label'] = name
                elif '-map' in name:
                    file_trace['train']['map'] = name
                elif '.json' in name:
                    file_trace['train']['json'] = name
                else:
                    file_trace['train']['set'] = name
            elif '-test-' in name:
                if '-label' in name:
                    file_trace['test']['label'] = name
                elif '-map' in name:
                    file_trace['test']['map'] = name
                elif '.json' in name:
                    file_trace['test']['json'] = name
                else:
                    file_trace['test']['set'] = name

        feature_size, class_size, crimes = extract_data(cur_zip, file_trace)

    # Create the model
    x = tf.placeholder(tf.float64, [None, feature_size])
    W = tf.Variable(tf.zeros([feature_size, class_size], dtype=tf.float64))
    b = tf.Variable(tf.zeros([class_size], dtype=tf.float64))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # y = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(x, W) + b, crimes.train.labels)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float64, [None, class_size])

    cross_entropy = tf.reduce_mean(
        # -tf.mul(tf.div(1.0, class_size), tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])
        # -tf.reduce_sum(y_ - y, reduction_indices=[1])
        # tf.mul(tf.div(-1.0, class_size), tf.reduce_sum(y_ + y, reduction_indices=[1])) # with 0.5
        # -tf.reduce_sum(y_ * tf.mul(y, 0.5), reduction_indices=[1])
        # -(y_ * tf.mul(y, 0.5))
        # -tf.reduce_sum(y_ * y, reduction_indices=[1])
        # -tf.mul(tf.div(1.0, class_size), tf.reduce_sum(y_ * tf.mul(y, 0.42), reduction_indices=[1]))
    )
    # .minimize = back propagation
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    print("> Train")
    # Train
    # tf.initialize_all_variables().run()
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(int(len(crimes.train) / 1000)):
            batch_xs, batch_ys = crimes.train.next_batch(1000)
            # print(">>", i, batch_xs, batch_ys, len(batch_xs), len(batch_ys))
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            # print("W:", sess.run(W))
            # print("b:", sess.run(b))
            # print("y", sess.run(y, feed_dict={x: batch_xs}))
            # input()

        print("W:", sess.run(W))
        print("b:", sess.run(b))
        print("y", sess.run(y, feed_dict={x: crimes.train.crimes}))

        print("> Test model")
        # Test trained model
        get_res = tf.argmax(y, 1)
        test_res = sess.run(get_res, feed_dict={x: crimes.test.crimes})

        get_correct_res = tf.argmax(y_, 1)
        test_res_correct = sess.run(
            get_correct_res, feed_dict={y_: crimes.test.labels})

        res_unique, res_counts = numpy.unique(test_res, return_counts=True)

        print("> Res.: {}".format(test_res))
        print(">> Classes: {}".format(res_unique))
        print(">>> Class count: {}".format(dict(zip(res_unique, res_counts))))

        res_corr_unique, res_corr_counts = numpy.unique(
            test_res_correct, return_counts=True)

        print("> Correct Res.: {}".format(test_res_correct))
        print(">> Classes: {}".format(res_corr_unique))
        print(">>> Class count: {}".format(
            dict(zip(res_corr_unique, res_corr_counts))))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # print(sess.run(correct_prediction, feed_dict={x: crimes.test.crimes, y_: crimes.test.labels}))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        res_accuracy = sess.run(
            accuracy, feed_dict={x: crimes.test.crimes, y_: crimes.test.labels})
        print("> Accuracy: {:.5f}".format(res_accuracy))

        logloss = -tf.mul(
            tf.div(tf.cast(1.0, tf.float64), tf.cast(
                len(crimes.test.crimes), tf.float64)),
            tf.reduce_sum(crimes.test.labels * tf.log(
                sess.run(y, feed_dict={x: crimes.test.crimes})
            ), [0, 1])
        )
        res_logloss = sess.run(logloss)
        print("> logloss: {:.5f}".format(res_logloss))

        gen_confusion_matrix(test_res, test_res_correct, class_size)


def gen_confusion_matrix(res, correct_res, classes, plot=True):
    matrix = []

    for class_ in range(classes):
        matrix.append([0 for _ in range(classes)])

    for num, val in enumerate(res):
        matrix[correct_res[num]][val] += 1

    # Not normalized
    # for row in matrix:
    #     print(row)

    # Normalize matrix
    for row in matrix:
        for pos in range(len(row)):
            row[pos] = float(row[pos] / len(res))

    # Normalized
    # for row in matrix:
    #     print(row)

    if plot:
        plot_confusion_matrix(matrix, classes)

def plot_confusion_matrix(matrix, classes):
    import matplotlib.pyplot as plt

    cmap = plt.cm.Blues
    title = 'Confusion matrix'

    numpy.set_printoptions(precision=2)

    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(classes)
    plt.xticks(tick_marks, [str(num) for num in range(classes)]) #, rotation=45)
    plt.yticks(tick_marks, [str(num) for num in range(classes)])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

