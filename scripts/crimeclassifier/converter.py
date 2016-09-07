from . utils import read_csv
from . utils import MsgLoad
from json import load
from os import path
from random import randint
from datetime import datetime
from struct import pack

__all__ = ['to_bin']


def extract_date(date, feature_data):
    """Convert Date in a vector normalized.

    Params:
        date (string): the date string
        feature_data (dict): information about the year. 'max_year', 'min_year'
                             and 'year_range'

    Returns:
        list of float
    """
    tmp = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    result = []

    result.append(
        (tmp.year - feature_data['min_year']) / feature_data['year_range'])
    result.append(tmp.month / 12.)
    result.append(tmp.day / 31.)
    result.append(tmp.hour / 24.)
    result.append(tmp.minute / 60.)

    return result


def row_to_vector(row, report, selected_features, fcn):
    """Convert a CSV row in a normalized vector.

    The order of the features is the same of the file config if
    given otherwise is the report order. Se to_bin function for more
    details

    Params:
        row (dict): current data
        report (dict): report of the current dataset
        selected_features (list): list of features to insert in the vector
        fnc (string): feature class name

    Returns
        list: a vector of normalized float values
    """
    values = []
    class_ = None

    for feature in selected_features:
        if report['features'][feature]['type'] == 'date':
            for val in extract_date(row[feature], report['features'][feature]):
                values.append(val)
        elif report['features'][feature]['type'] == 'number':
            values.append(
                (float(row[feature]) - report['features'][feature]['min']) /
                report['features'][feature]['range'])
        elif report['features'][feature]['type'] == 'string':
            if feature != fcn:
                values.append(
                    report['features'][feature]['p_vals'][row[feature]])
            else:
                class_ = report['features'][feature]['p_vals'][row[feature]]

    return (values, class_)


def get_fix_range_value(tot_elms, index):
    """Percentage value of the given index in a set of elements.

    The value returned is in the middle:

    For example:

    0.0               1.0
     |-----|-----|-----|
        ^     ^     ^

    Params:
        tot_elems (int): number of elements in the set
        index (int): the index of the current element,
                     starting from 1

    Return:
        float: percentage with dot notation,
               so from 0.0 to 1.0

    """
    step = (1. / tot_elms) / 2.
    return float((index / tot_elms) - step)


def write_binary(data_filename, label_filename, map_filename,
                 num_elems, num_classes, bucket):
    """Put data into binary format.

    Params:
        data_filename (string): name of the data file to write
        label_filename (string): name of the label file to write
        map_filename (string): name of the map file to write
        num_elems (int): number of elements of current set
        num_classes (int): number of the current classes
        bucket (list): list of records
    """

    msg_load = MsgLoad()
    binary_index = 0  # counter

    with open(data_filename, "wb") as data_f:

        data_f.write(pack('>I', 3584 + 1))  # double
        data_f.write(pack('>I', num_elems))
        data_f.write(pack('>I', len(bucket[0][1][0])))

        with open(label_filename, "wb") as label_f:

            label_f.write(pack('>I', 3072 + 1))  # integer
            label_f.write(pack('>I', num_elems))
            # num of classes
            label_f.write(pack('>I', num_classes))

            with open(map_filename, "wb") as map_f:

                map_f.write(pack('>I', 3072 + 1))  # integer
                map_f.write(pack('>I', num_elems))
                # binary index, csv index
                map_f.write(pack('>I', 2))

                while binary_index < num_elems:
                    csv_index, (cur_features, cur_class) = bucket.pop(
                        randint(0, len(bucket) - 1))

                    data_f.write(
                        pack('d' * len(cur_features), *cur_features))

                    label_f.write(pack('I', cur_class))

                    map_f.write(pack('II', binary_index, csv_index))

                    binary_index += 1

                    msg_load.show(
                        "--> Writed {} of {}".format(binary_index, num_elems))

    msg_load.show("--> Writed {} of {}".format(binary_index, num_elems))


def to_bin(report_f, config_f):
    """Convert a dataset from CSV to binary.

    Report have to be a JSON file computed by the analyzer. The config file
    have to be in JSON format and has the following keys:

    {
        out_filename: the target name of the converted dataset
        train_percentage: integer from 1 to 99
        features: list of the names of the features to extract
        feature_class_name: name of the feature that represent the class of a
                            record
        class_filter: list of classes to filter (extract)
    }

    Params:
        report_f (string): report filename
        config_f (string): config filename
    """

    msg_load = MsgLoad()

    print("-> Open report file")
    with open(report_f, 'r') as o_f:
        report = load(o_f)

    print("-> Open config file")
    with open(config_f, 'r') as o_f:
        config = load(o_f)

    print("-> Select features")
    if len(config.get('features', [])) == 0:
        selected_features = report['features'].keys()
    else:
        selected_features = config.get('features')

    print("-> Check feature class name")
    if config.get('feature_class_name', '') not in selected_features:
        raise Exception(
            "Feature class name not valid or not exists")
    else:
        fcn = config.get('feature_class_name')

    print("-> Generate class filter")
    if len(config.get('class_filter', [])) != 0:
        class_filter = [
            name for name, count in report['features'][fcn]['set']
            if name in config.get('class_filter')]
    else:
        class_filter = [
            name for name, count in report['features'][fcn]['set']]

    print("-> Resize feature class")
    report['features'][fcn]['set'] = [
        (name, count) for name, count in report['features'][fcn]['set']
        if name in class_filter]

    for feature in selected_features:
        if report['features'][feature]['type'] == 'string':
            report['features'][feature]['p_vals'] = {}
            if feature != fcn:
                if config.get('scaled_sets', False):
                    ##
                    # Is not the maximum but the middle of the range
                    #
                    # For example:
                    #
                    # 0.0                 1.0
                    #  |-----|---------|---|
                    #     ^       ^      ^
                    #
                    cur_val = report['num_records']
                    for name, count in report['features'][feature]['set']:
                        report['features'][feature]['p_vals'][name] = float(
                            (cur_val - count / 2) / report['num_records'])
                        cur_val -= count
                else:
                    ##
                    # Like the scaled solution but with fixed
                    # ranges
                    #
                    # For example:
                    #
                    # 0.0               1.0
                    #  |-----|-----|-----|
                    #     ^     ^     ^
                    tot_elms = len(report['features'][feature]['set'])
                    for index, (name, count) in enumerate(reversed(
                            report['features'][feature]['set']), 1):
                        report['features'][feature]['p_vals'][
                            name] = get_fix_range_value(tot_elms, index)
            else:
                for index, (name, count) in enumerate(
                        report['features'][feature]['set']):
                    report['features'][feature]['p_vals'][name] = index

            ##
            # DEBUG
            # print(list(reversed(
            #     sorted(report['features'][feature]['p_vals'].items(),
            #            key=lambda elm: elm[1]))))

    crimes = []

    print("-> Extract records")
    for num, row in enumerate(read_csv(report.get('filename')), 1):
        msg_load.show(
            "--> Parsed {} of {}".format(num, report.get('num_records')))

        if row[fcn] in class_filter:
            crimes.append(
                ##
                # (csv_index, ([features], class))
                (num, row_to_vector(row, report, selected_features, fcn)))

    print("--> Parsed {} of {}".format(num, report['num_records']))

    tot_crimes_available = len(crimes)
    train_num = int(
        tot_crimes_available / 100.0 * config.get("train_percentage", 80.))
    test_num = tot_crimes_available - train_num

    print("-> Tot crimes available: {}".format(tot_crimes_available))

    dataset_folder = path.dirname(report['filename'])

    print("-> Generate train set")

    train_filename = path.join(
        dataset_folder, "{}-train-crimes".format(config['out_filename']))
    train_label_filename = path.join(
        dataset_folder, "{}-train-crimes-label".format(config['out_filename']))
    train_map_filename = path.join(
        dataset_folder, "{}-train-crimes-map".format(config['out_filename']))

    write_binary(train_filename, train_label_filename, train_map_filename,
                 train_num, len(crimes[0][1][0]), crimes)

    print("-> Generate test set")

    test_filename = path.join(
        dataset_folder, "{}-test-crimes".format(config['out_filename']))
    test_label_filename = path.join(
        dataset_folder, "{}-test-crimes-label".format(config['out_filename']))
    test_map_filename = path.join(
        dataset_folder, "{}-test-crimes-map".format(config['out_filename']))

    write_binary(test_filename, test_label_filename, test_map_filename,
                 test_num, len(crimes[0][1][0]), crimes)
