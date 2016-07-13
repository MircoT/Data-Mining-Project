from os import path
from io import TextIOWrapper
from zipfile import ZipFile
from csv import DictReader

__all__ = ["readCSV"]

def readCSV(filename):
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
                for row in DictReader(TextIOWrapper(data_f, newline=''), delimiter=','):
                    yield row
                
    elif extension == '.csv':
        with open(filename, 'r', newline='') as data_f:
            for row in DictReader(data_f, delimiter=','):
                yield row