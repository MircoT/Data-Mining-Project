from . utils import read_csv
from . utils import MsgLoad
from json import dump

__all__ = ['create_report']


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

    msg_load = MsgLoad()

    report = {
        'features': {},
        'num_records': 0
    }

    print("-> Open CSV file")

    for row in read_csv(filename):
        if len(report['features']) == 0:
            print("-> Read features")
            for feature in row.keys():
                report['features'][feature] = {
                    'type': None
                }

        for feature, value in row.items():
            if report['features'][feature]['type'] is None:
                try:
                    float(value)
                    report['features'][feature]['type'] = 'number'
                    report['features'][feature]['max'] = -10**1000
                    report['features'][feature]['min'] = 10**1000
                except ValueError:
                    report['features'][feature]['type'] = 'string'
                    report['features'][feature]['set'] = {}

            if report['features'][feature]['type'] == 'number':
                cur_val = float(value)
                if cur_val > report['features'][feature]['max']:
                    report['features'][feature]['max'] = cur_val
                elif cur_val < report['features'][feature]['min']:
                    report['features'][feature]['min'] = cur_val
            elif report['features'][feature]['type'] == 'string':
                if value not in report['features'][feature]['set']:
                    report['features'][feature]['set'][value] = 0
                report['features'][feature]['set'][value] += 1

        report['num_records'] += 1

        msg_load.show("-> Parsed {} records".format(report['num_records']))

    print("-> Parsed {} records {}".format(
        report['num_records'],
        "..."
    ))

    for num, (feature, details) in enumerate(report['features'].items(), 1):
        if details['type'] == 'number':
            details['range'] = details['max'] - details['min']
        elif details['type'] == 'string':
            details['set'] = list(
                reversed(
                    sorted(
                        [(key, value)
                         for key, value in details['set'].items()],
                        key=lambda elm: elm[1]
                    )
                )
            )
            details['len'] = len(details['set'])

        msg_load.show("-> Analyzed {} features".format(num))

    print("-> Analyzed {} features {}".format(
        len(report['features']),
        "..."
    ))

    if export:
        print("-> Write report.json file")
        with open('report.json', 'w') as report_file:
            dump(report, report_file, indent=2)

    print("-> Report done!")

    return report
