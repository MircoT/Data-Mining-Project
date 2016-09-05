from . utils import read_csv
from . utils import MsgLoad
from json import dump
from time import time
from os.path import basename
from datetime import datetime

__all__ = ['create_report']


def to_list_sorted_by_value(dictionary):
    """Convert a dictionary in a sorted list by value."""
    return list(
        reversed(
            sorted(
                [(key, value)
                 for key, value in dictionary.items()],
                key=lambda elm: elm[1]
            )
        )
    )


def check_value_type(value):
    """Verify the type of a value in a csv.

    Params:
        value (string): the data string

    Returns:
        dict {
            type: the type of the value

            max, min -> if type is number
            max_year, min_year -> if type is date
            set -> if type is string
        }
    """
    try:
        float(value)
        return {
            'type': "number",
            'max': -10**1000,
            'min': 10**1000
        }
    except:
        pass
    try:
        datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return {
            'type': "date",
            'max_year': -10**1000,
            'min_year': 10**1000
        }
    except:
        pass

    return {
        'type': "string",
        'set': {}
    }


def create_report(filename, export=False):
    """Analyze and create a report of a CSV file.

    This function returns the report dictionary and
    optionally creates 'a report.json' file.

    Params:
        filename (string): name of the file that contains the data
        export (boolean): if True the function will export a report.json

    Returns:
        report (dict)
    """

    start = time()

    msg_load = MsgLoad()

    report = {
        'features': {},
        'num_records': 0,
        'filename': filename
    }

    print("> Open CSV file")

    for row in read_csv(filename):
        if len(report['features']) == 0:
            print("> Read features")
            for feature in row.keys():
                report['features'][feature] = {
                    'type': None
                }

        for feature, value in row.items():
            if report['features'][feature]['type'] is None:
                report['features'][feature].update(
                    check_value_type(value).items())

            if report['features'][feature]['type'] == 'number':
                cur_val = float(value)
                if cur_val > report['features'][feature]['max']:
                    report['features'][feature]['max'] = cur_val
                if cur_val < report['features'][feature]['min']:
                    report['features'][feature]['min'] = cur_val
            elif report['features'][feature]['type'] == 'date':
                cur_year = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").year
                if cur_year > report['features'][feature]['max_year']:
                    report['features'][feature]['max_year'] = cur_year
                elif cur_year < report['features'][feature]['min_year']:
                    report['features'][feature]['min_year'] = cur_year
            elif report['features'][feature]['type'] == 'string':
                if value not in report['features'][feature]['set']:
                    report['features'][feature]['set'][value] = 0
                report['features'][feature]['set'][value] += 1

        report['num_records'] += 1

        msg_load.show("> Parsed {} records".format(report['num_records']))

    print("> Parsed {} records {}".format(
        report['num_records'],
        "..."
    ))

    for num, (feature, details) in enumerate(report['features'].items(), 1):
        if details['type'] == 'number':
            details['range'] = details['max'] - details['min']
        elif details['type'] == 'date':
            details['year_range'] = details['max_year'] - details['min_year']
        elif details['type'] == 'string':
            details['set'] = to_list_sorted_by_value(details['set'])
            details['len'] = len(details['set'])

        msg_load.show("> Analyzed {} features".format(num))

    print("> Analyzed {} features {}".format(
        len(report['features']),
        "..."
    ))

    if export:
        print("-> Write {}_report.json file".format(basename(filename)))
        with open('{}_report.json'.format(
                basename(filename)), 'w') as report_file:
            dump(report, report_file, indent=2)

    print("-> Report done in {:0.3f} sec.".format(time() - start))

    return report
