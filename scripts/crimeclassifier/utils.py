# -*- coding: utf-8 -*-
from os import path
from io import TextIOWrapper
from zipfile import ZipFile
from csv import DictReader
from time import time

__all__ = ['read_csv', 'MsgLoad']


class MsgLoad(object):

    """Print a cool loading on console."""

    def __init__(self):
        self.loading_bars = ['▄', '█', '▀', '█']
        self.loading_counter = 0
        self.start_time = time()

    def show(self, message):
        """Show the message with the loading spin.

        Args:
            message (string): the text to print
        """

        if time() - self.start_time >= 0.42:
            print("{} {}".format(
                message, self.loading_bars[self.loading_counter]), end='\r')
            self.start_time = time()
            self.loading_counter = (self.loading_counter + 1) % 4


def read_csv(filename):
    """Extract the data from a CSV file format.

    The data file can be in zip format or a plain CSV. This
    function will return a generator that iterate over each
    row of the CSV file. Rows are parsed with csv.DictReader
    so you will have a dict for each row data.

    Params:
        filename (string): name of the file that contains the data.

    Returns:
        generator: data extracted from the CSV file.
    """
    if not path.exists(filename):
        raise Exception("Error: file not exist!")

    real_filename, extension = path.splitext(filename)

    if extension == '.zip':
        with ZipFile(filename) as zip_f:
            with zip_f.open(path.basename(real_filename), 'r') as data_f:
                for row in DictReader(
                        TextIOWrapper(data_f, newline=''),
                        delimiter=','):
                    yield row

    elif extension == '.csv':
        with open(filename, 'r', newline='') as data_f:
            for row in DictReader(data_f, delimiter=','):
                yield row
