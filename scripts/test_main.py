from crimeclassifier import *


def main():
    # create_report("../datasets/train.csv.zip", True)
    to_bin("train.csv.zip_report.json", "config.json")


if __name__ == '__main__':
    main()
