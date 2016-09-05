from . utils import read_csv
from . utils import MsgLoad
from json import load
from os.path import dirname

__all__ = ['to_bin']


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

    if config.get('feature_class_name', '') not in report['features']:
        raise Exception(
            "Feature class name not valid or not exists")

    crimes = []

    ## ----- TO DO -----
    # for num, row in enumerate(read_csv(filename), 1):
    #     msg_load.show("> Parsed {} of {}".format(num, report['num_records']))

    #     if row[class_name] in classes:
    #         cur_crime_obj = Record(
    #             row, report['features'], features, class_name)

    #         ##
    #         # print(cur_crime_obj)

    #         crimes.append(cur_crime_obj)

    # print("> Parsed {} of {}".format(num, report['num_records']))
