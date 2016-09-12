# Data-Mining-Project
An approach to data mining solving a Kaggle problem with TensorFlow.

## Problem reference

[Predict the category of crimes that occurred in the city by the bay](https://www.kaggle.com/c/sf-crime)

## Kaggle Dataset

Kaggle is hosting this competition for the machine learning community to use for fun and practice.
This dataset is brought to you by SF OpenData, the central clearinghouse for data published by the
City and County of San Francisco.

The dataset is available in this repository in the `dataset` folder (`train.csv.zip` file).

## Dependencies

The scripts are created using *Python 3* and *TensorFlow* **r0.10**. You have also to install these libraries:

* [TensorFlow](https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html#download-and-setup)
* [NumPy](http://www.scipy.org/scipylib/download.html)
* [Matplotlib](http://matplotlib.org/users/installing.html) 

## How To use it

The report of the crime dataset is already in the script folder. If you want to create a report like that you can use the proper command:

```
cd scripts
python crimeclassifier.py genReport ../datasets/train.csv.zip
```

To generate the binary file useful for the classification:

```
# From scripts folder
python crimeclassifier.py genBin train.csv.zip_report.json config_examples/config_with_class_filter.json
```

To proceed with the classification of the binary generated with the previous command use the following lines:

```
# From scripts folder
python crimeclassifier.py classify ../datasets/Filtered_dataset.zip 
```

## TensorBoard

The classify command generate automatically a log folder to open with TensorBoard. Results of each command
executed will be stored inside this log folder, so if you want a clean report delete lof folder first.

```
# From scripts folder
tensorboard --logdir ./log   
```

## Example config file

```
{
    "out_filename": "example_output_name",
    "train_percentage": 75,
    "features": [
        "Dates",
        "Category",
        "DayOfWeek",
        "X",
        "Y"
    ],
    "feature_class_name": "Category",
    "class_filter": [],
    "scaled_sets": true
}
```

### Notes

* out_filename: the name of the zip file generated during genBin command
* train_percentage: the size in percentage of the train set, the remaining is for the test set
* features: dataset features to extract
* feature_class_name: must be present in features list
* class_filter: list of strings that are present in the feature_class_name, otherwise they are taken all
* scaled_sets: scale the values of the classes found

