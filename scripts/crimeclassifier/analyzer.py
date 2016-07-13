from . utils import read_csv
from json import dump
from time import time

__all__ = ['create_report']


def create_report(filename):
    """Analyze and create a report of a CSV file.

    This function returns nothing but creates 'a report.json' file.

    Params:
        filename (string): name of the file that contains the data
    """

    report = {
        'features': {},
        'num_records': 0
    }

    loading_bars = ['▙', '▛', '▜', '▟']
    loading_counter = 0
    start_time = time()

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

        if time() - start_time >= 0.42:
            print("-> Parsed {} records {}".format(
                report['num_records'],
                loading_bars[loading_counter]
            ), end='\r')
            start_time = time()
            loading_counter = (loading_counter + 1) % 4

    print("-> Parsed {} records {}".format(
        report['num_records'],
        "..."
    ))

    start_time = time()

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

        if time() - start_time >= 0.42:
            print("-> Analyzed {} features {}".format(
                num,
                loading_bars[loading_counter]
            ), end='\r')
            start_time = time()
            loading_counter = (loading_counter + 1) % 4

    print("-> Analyzed {} features {}".format(
        len(report['features']),
        "..."
    ))

    print("-> Write report.json file")
    with open('report.json', 'w') as report_file:
        dump(report, report_file, indent=2)

    print("-> Done!")
