from . utils import read_csv
from . utils import MsgLoad
from json import load
from json import dump
from os import path
from os import remove
from random import randint
from datetime import datetime
from struct import pack
import zipfile

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


def write_binary(data_filename, label_filename, map_filename, stat_filename,
                 num_elems, num_classes, bucket, class_list):
    """Put data into binary format.

    Params:
        data_filename (string): name of the data file to write
        label_filename (string): name of the label file to write
        map_filename (string): name of the map file to write
        stat_filename (string): name of the stats file to write
        num_elems (int): number of elements of current set
        num_classes (int): number of the current classes
        bucket (list): list of records

    File format:

        +-------------------+
        |magic number       |
        |size in dimension 0|
        |size in dimension 1|
        |size in dimension 2|
        |       .....       |
        |size in dimension N|
        |data               |
        +-------------------+

        The magic number is an integer (MSB first).
        The first 2 bytes are always 0.

        The third byte codes the type of the data:
        0x08: unsigned byte
        0x09: signed byte
        0x0B: short (2 bytes)
        0x0C: int (4 bytes)
        0x0D: float (4 bytes)
        0x0E: double (8 bytes)

        The 4-th byte codes the number of dimensions of the vector/matrix:
        1 for vectors, 2 for matrices....

        The sizes in each dimension are 4-byte integers (MSB first,
        high endian, like in most non-Intel processors).

        The data is stored like in a C array.

        ----- CRIMES FILE -----

        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000E01(3584) magic number
        0004     32 bit integer  ??               number of items
        0008     32 bit integer  ??               number of features x crime
        0010     64 bit double   ??               crime feature
        0018     64 bit double   ??               crime feature
        ........
        xxxx     64 bit double   ??               crime feature


        ----- LABEL FILE -----

        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000C01(3073) magic number (MSB first)
        0004     32 bit integer  ??               number of items
        0008     32 bit integer  ??               number classes
        000C     32 bit integer  ??               label
        0010     32 bit integer  ??               label
        ........
        xxxx     32 bit integer  ??               label


        ----- MAP FILE -----

        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000C01(3073) magic number (MSB first)
        0004     32 bit integer  ??               number of items
        0008     32 bit integer  0x00000002   (2) number values
        000C     32 bit integer  ??               binary index
        0010     32 bit integer  ??               csv index
        0014     32 bit integer  ??               binary index
        0018     32 bit integer  ??               csv index
        ........
        xxxx     32 bit integer  ??               xxx
    """

    msg_load = MsgLoad()
    binary_index = 0  # counter

    stats = {
        'list': class_list
    }

    with open(data_filename, "wb") as data_f:

        data_f.write(pack('>I', 3584 + 1))  # double
        data_f.write(pack('>I', num_elems))
        data_f.write(pack('>I', len(bucket[0][1][0])))

        with open(label_filename, "wb") as label_f:

            label_f.write(pack('>I', 3072 + 1))  # integer
            # num records
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

                    if class_list[cur_class] not in stats:
                        stats[class_list[cur_class]] = 0
                    stats[class_list[cur_class]] += 1

                    data_f.write(
                        pack('d' * len(cur_features), *cur_features))

                    label_f.write(pack('I', cur_class))

                    map_f.write(pack('II', binary_index, csv_index))

                    binary_index += 1

                    msg_load.show(
                        "--> Writed {} of {}".format(binary_index, num_elems))

    with open(stat_filename, 'w') as stat_file:
        dump(stats, stat_file, indent=2)

    print("--> Writed {} of {} √".format(binary_index, num_elems))


def gen_f_name(dataset_path, name, set_):
    """Create all filename fot a kind of set.

    The path will be relative of the report dataset_path.

    Params:
        dataset_path (string): the dataset path in the report
        name (string): the target name
        set_ (string): the type of the set

    Example:
        name = foo
        set_ = bar

        will generate:

        - foo-bar-crimes
        - foo-bar-crimes-label
        - foo-bar-crimes-map
        - foo-bar-crimes.json
    """
    return [
        path.join(
            path.dirname(dataset_path),
            "{}-{}-crimes{}".format(name, set_, type_)
        )
        for type_ in ['', '-label', '-map', '.json']
    ]


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

    print("--> Parsed {} of {} √".format(num, report['num_records']))

    train_percentage = config.get("train_percentage", 80.)
    tot_crimes_available = len(crimes)
    train_num = int(
        tot_crimes_available / 100.0 * train_percentage)
    test_num = tot_crimes_available - train_num

    print("-> Tot crimes available: {}".format(tot_crimes_available))

    dataset_folder = path.dirname(report['filename'])

    print("-> Generate train set [{:0.2f}% of {}]".format(
        train_percentage, tot_crimes_available))

    train_filenames = gen_f_name(
        report['filename'], path.relpath(config['out_filename']), 'train')

    write_binary(*train_filenames,
                 train_num, len(class_filter), crimes,
                 [name for name, count in report['features'][fcn]['set']])

    print("-> Generate test set [{:0.2f}% of {}]".format(
        100. - train_percentage, tot_crimes_available))

    test_filenames = gen_f_name(
        report['filename'], path.relpath(config['out_filename']), 'test')

    write_binary(*test_filenames,
                 test_num, len(class_filter), crimes,
                 [name for name, count in report['features'][fcn]['set']])

    print("-> Create zip file")

    with zipfile.ZipFile(path.join(
        path.dirname(report['filename']),
        "{}.zip".format(config['out_filename'])
    ), "w", zipfile.ZIP_DEFLATED) as zip_file:

        for file_ in train_filenames + test_filenames:
            print(" "*70, end='\r')
            print("--> Zip {}".format(file_), end='\r')
            zip_file.write(file_, arcname=path.basename(file_))
            remove(file_)

    print(" "*70, end='\r')
    print("-> Zip file Done")
