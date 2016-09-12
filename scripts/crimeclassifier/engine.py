from collections import namedtuple
from io import BytesIO
import json
import zipfile
import numpy

##
# Tensorflow
from tensorflow.python.framework import dtypes
import tensorflow as tf


class DataSet(object):

    """Object to manage the dataset."""

    def __init__(self,
                 crimes,
                 labels,
                 dtype=dtypes.float64):
        """Initialize manager object.

        Arguments:
            crimes (numpy vector): data records
            labels (numpy vector): labels

        Keyword Arguments:
            dtype (dtypes): type of the records (default: {dtypes.float64})
        """

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32, dtypes.float64):
            raise TypeError(
                'Invalid crime dtype {}, expected uint8 or float32 or float64'.format(dtype))

        assert crimes.shape[0] == labels.shape[0], (
            'crimes.shape: %s labels.shape: %s' % (crimes.shape, labels.shape))
        self._num_examples = crimes.shape[0]

        if dtype == dtypes.float32:
            crimes = crimes.astype(numpy.float32)
        elif dtype == dtypes.float64:
            crimes = crimes.astype(numpy.float64)

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
    """Read an integer from a byte stream.

    Arguments:
        bytestream (bytes): stream of bytes to read

    Returns:
        (unsigned int 32): readed number
    """
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_records(zip_file, filename):
    """Get all labels.

    Arguments:
        zip_file (zipfile.ZipFile): the zipfile obj
        filename (string): the name of data file

    Returns:
        (int, numpy vector): number of the features and vector with data

    Raises:
        ValueError: if the magic number is not valid for this type of file
    """
    print('-> Extracting data [{}] '.format(filename))
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
    return num_features, data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors.

    Arguments:
        labels_dense (numpy vector): the list of labels
        num_classes (int): number of all classes

    Returns:
        numpy vector -- one hot vector with the labels passed
    """
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(zip_file, filename):
    """Get all record labels.

    Arguments:
        zip_file (zipfile.ZipFile): the zipfile obj
        filename (string): the name of label file

    Returns:
        (int, numpy vector): number of the classes and one hot vector
                             with the labels

    Raises:
        ValueError: if the magic number is not valid for this type of file
    """
    print('-> Extracting label [{}] '.format(filename))
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
    """Get records data and labels.

    Params:
        zip_file (zipfile.ZipFile): the zipfile obj
        files (dict): names of the files for train and test sets

    Returns:
        (int, int, namedtuple): you will have the feature size, the
                                class size and the train set with the
                                test set in a namedtuple
    """
    Datasets = namedtuple('Datasets', ['train', 'test'])

    feature_size_train, train_crimes = extract_records(
        zip_file, files['train']['set'])
    class_size_train, train_labels = extract_labels(
        zip_file, files['train']['label'])

    feature_size_test, test_crimes = extract_records(
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


def classify(filename, plot=True):

    print("-> Open zip file [{}]".format(filename))

    with zipfile.ZipFile(filename) as cur_zip:
        file_trace = {
            'train': {},
            'test': {}
        }

        print("-> Check file names")
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

        print("-> Extract all")
        feature_size, class_size, crimes = extract_data(cur_zip, file_trace)

        print("-> Open stat file")
        with cur_zip.open(file_trace['train']['json']) as stat_f:
            stats = json.loads(stat_f.read().decode('utf-8'))

    # Create the model
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float64, [None, feature_size], name="x")

    with tf.name_scope("weights"):
        W = tf.Variable(
            tf.zeros([feature_size, class_size], dtype=tf.float64), name="W")

    with tf.name_scope("biases"):
        b = tf.Variable(tf.zeros([class_size], dtype=tf.float64), name="b")

    with tf.name_scope("softmax"):
        y = tf.nn.softmax(tf.matmul(x, W) + b, name="y")

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float64, [None, class_size], name="y_")

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]),
            name="reduce_mean"
        )

    with tf.name_scope('train'):
        # .minimize = back propagation
        train_step = tf.train.GradientDescentOptimizer(
            0.5, name="GDO").minimize(cross_entropy)

    with tf.name_scope('Accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # create a summary for our cost and accuracy
    tf.scalar_summary("cost", cross_entropy)
    tf.scalar_summary("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can execute in a
    # session
    summary_op = tf.merge_all_summaries()

    print("@-> Train")

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        ##
        # Create symmary
        s_writer = tf.train.SummaryWriter("./log", graph=tf.get_default_graph())

        for i in range(int(len(crimes.train) / 1000)):
            batch_xs, batch_ys = crimes.train.next_batch(1000)

            ##
            # Training output
            # print("########## I[{}] ##########".format(i))
            # print("----- [batch xs] -----\n{}\n LEN({})".format(batch_xs,
            #                                                     len(batch_xs)))
            # print("----- [batch ys] -----\n{}\n LEN({})".format(batch_ys,
            #                                                     len(batch_ys)))

            _, summary = sess.run(
                [train_step, summary_op],
                feed_dict={x: batch_xs, y_: batch_ys})
            s_writer.add_summary(summary, i)
            ##
            # Training output
            # print("----- [Matrix W] -----\n{}".format(sess.run(W)))
            # print("----- [Vector b] -----\n{}".format(sess.run(b)))
            # print("----- [Vector y] -----\n{}".format(
            #     sess.run(y, feed_dict={x: batch_xs})))
            # input()

        print("----- [Matrix W] -----\n{}".format(sess.run(W)))
        print("----- [Vector b] -----\n{}".format(sess.run(b)))
        print("----- [Vector y] -----\n{}".format(
            sess.run(y, feed_dict={x: crimes.train.crimes})))

        print("@-> Test trained model")

        get_res = tf.argmax(y, 1)
        test_res = sess.run(get_res, feed_dict={x: crimes.test.crimes})

        get_correct_res = tf.argmax(y_, 1)
        test_res_correct = sess.run(
            get_correct_res, feed_dict={y_: crimes.test.labels})

        res_unique, res_counts = numpy.unique(test_res, return_counts=True)

        print("@-> Train Results: \n{}".format(test_res))
        print("--> Train Result Classes:\n{}".format(res_unique))
        print("--> Train Result Class count:\n{}".format(
            dict(zip(res_unique, res_counts))))

        res_corr_unique, res_corr_counts = numpy.unique(
            test_res_correct, return_counts=True)

        print("@-> TRUE Results: \n{}".format(test_res_correct))
        print("--> TRUE Result Classes:\n{}".format(res_corr_unique))
        print("--> TRUE Result Class count:\n{}".format(
            dict(zip(res_corr_unique, res_corr_counts))))

        print("@-> Correct prediction: \n{}".format(
            sess.run(correct_prediction,
                     feed_dict={x: crimes.test.crimes,
                                y_: crimes.test.labels})))
        res_accuracy = sess.run(
            accuracy, feed_dict={
                x: crimes.test.crimes, y_: crimes.test.labels})
        print("!!-> [Accuracy] = {:.5f}".format(res_accuracy))

        logloss = -tf.mul(
            tf.div(tf.cast(1.0, tf.float64), tf.cast(
                len(crimes.test.crimes), tf.float64)),
            tf.reduce_sum(crimes.test.labels * tf.log(
                sess.run(y, feed_dict={x: crimes.test.crimes})
            ), [0, 1])
        )
        res_logloss = sess.run(logloss)
        print("!!-> [Logloss] = {:.5f}".format(res_logloss))

        confusion_matrix = gen_confusion_matrix(
            test_res, test_res_correct, stats['list'])

        ##
        # Add image to TensorBoard
        if confusion_matrix is not list:
            image = tf.image.decode_png(
                confusion_matrix.getvalue(), channels=4)
            # Add the batch dimension
            image = tf.expand_dims(image, 0)
            # Add image to summary
            add_image_op = tf.image_summary("Confusion Matrix", image)
            image_summary = sess.run(add_image_op)
            s_writer.add_summary(image_summary)


def gen_confusion_matrix(res, correct_res, classes,
                         normalized=True, plot=True):
    """Create confusion matrix from results.

    Params:
        res (list): result vector
        correct_res (list): correct result vector
        classes (list of string): the class list
        normalized (bool): normalize the results or not
        plot (bool): show the results with matplotlib or not

    Returns:
        (list of lists): the result matrix
    """
    matrix = []

    for class_ in classes:
        matrix.append([0 for _ in range(len(classes))])

    for num, val in enumerate(res):
        matrix[correct_res[num]][val] += 1

    if normalized:
        for row in matrix:
            for pos in range(len(row)):
                row[pos] = float(row[pos] / len(res))

    if plot:
        return plot_confusion_matrix(matrix, classes)
    else:
        print("------- Confusion Matrix -------")
        for row in matrix:
            print(row)
        print("----- Confusion Matrix End -----")

    return matrix


def plot_confusion_matrix(matrix, classes):
    """Show on screen the confusion matrix.

    Params:
        matrix (list of lists): the matrix to plot
        classes (list of string): the class list
    """

    ##
    # Import plotter here because is not needed
    # for the classification
    import matplotlib.pyplot as plt

    cmap = plt.cm.Purples
    title = 'Confusion matrix'

    numpy.set_printoptions(precision=2)

    plt.figure()
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Binary for TensorBoard
    buf = BytesIO()
    plt.savefig(
        buf, format='png', dpi=100, pad_inches=2.0, bbox_inches='tight')
    buf.seek(0)

    plt.show()

    return buf
