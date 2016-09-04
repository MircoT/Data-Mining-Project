from . utils import read_csv
from . utils import MsgLoad
from json import load

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
    print("-> Open report file")
    with open(report_f, 'r') as o_f:
        report = load(o_f)
    print("-> Open config file")
    with open(config_f, 'r') as o_f:
        config = load(o_f)

    print(report)
    print(config)
