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


def new_dict_filtered_by_value
(dictionary):
    """Delete all keys that have 2 or less records."""
    return dict((key, value) for key, value in dictionary.items() if value > 2)





def report_crime_filter(report, export=False):
    """Filter the date and the addresses in the report."""
    msg_load = MsgLoad()

    print("> Filter dates")

    report['features']['Dates']['type'] = 'date'
    del report['features']['Dates']['len']

    report['features']['Dates']['yyyy'] = {
        'max': -10**1000,
        'min': 10**1000
    }

    for elm, num in report['features']['Dates']['set']:
        cur_date = extract_date(elm)
        if cur_date.yyyy > report['features']['Dates']['yyyy']['max']:
            report['features']['Dates']['yyyy']['max'] = cur_date.yyyy
        if cur_date.yyyy < report['features']['Dates']['yyyy']['min']:
            report['features']['Dates']['yyyy']['min'] = cur_date.yyyy

        msg_load.show("> Extracting date ranges")

    print("> Extracting date ranges ...")

    report['features']['Dates']['yyyy']['range'] = report['features'][
        'Dates']['yyyy']['max'] - report['features']['Dates']['yyyy']['min']

    del report['features']['Dates']['set']

    print("> Filter addresses - first phase")

    new_addresses = {}

    for elm, num in report['features']['Address']['set']:

        msg_load.show("> Extracting addresses")

        new_addr = filter_address(elm)

        if type(new_addr) == list:
            new_addr = elm

        if new_addr not in new_addresses:
            new_addresses[new_addr] = 0
        new_addresses[new_addr] += 1

    report['features']['Address'][
        'set'] = to_list_sorted_by_value(new_addresses)
    report['features']['Address']['len'] = len(
        report['features']['Address']['set'])

    print("> Extracting addresses ...")

    print("> Filter addresses - second phase")

    new_addresses = {}

    for elm, num in report['features']['Address']['set']:

        msg_load.show("> Extracting addresses phase two")

        new_addr = filter_address(elm)

        if type(new_addr) == list:
            try:
                first_addr = report['features'][
                    'Address']['set'].index(new_addr[0])
            except ValueError:
                first_addr = len(report['features']['Address']['set'])
            try:
                second_addr = report['features'][
                    'Address']['set'].index(new_addr[1])
            except ValueError:
                second_addr = len(report['features']['Address']['set'])

            if first_addr < second_addr:
                new_addr = new_addr[0]
            else:
                new_addr = new_addr[1]

        if new_addr not in new_addresses:
            new_addresses[new_addr] = 0
        new_addresses[new_addr] += 1

    # delete records with 2 as value
    len_before = len(new_addresses)
    new_addresses = new_dict_filtered_by_value(new_addresses)
    len_after = len(new_addresses)
    # common category
    new_addresses['OTHERS'] = len_before - len_after

    msg_load.show("> Extracting addresses phase two")

    report['features']['Address'][
        'set'] = to_list_sorted_by_value(new_addresses)
    report['features']['Address']['len'] = len(
        report['features']['Address']['set'])

    print("> Extracting addresses phase two ...")

    if export:
        print("> Write filter_report.json file")
        with open('filter_report.json', 'w') as filtered_file:
            dump(report, filtered_file, indent=2)

    print("> Report done!")

    return report


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
