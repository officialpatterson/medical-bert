from optparse import OptionParser

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(admissions, ratio):

    # Do some limited preprocessing
    X = admissions[['HADM_ID', 'text']]
    y = admissions['readm_30d']

    # Create a stratified train test split to preserver distribution.
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=ratio)
    test = pd.merge(X_test, y_test, left_index=True, right_index=True)
    train = pd.merge(X_train, y_train, left_index=True, right_index=True)

    return train, test


def main(input_data, train_out, test_out, ratio):

    # read the dataset from file.
    admissions = pd.read_csv(input_data)

    # split into training and testing
    train, test = split_data(admissions, ratio)

    # now save the files
    train.to_csv(train_out, index=None)
    test.to_csv(test_out, index=None)

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--input", help="specify the input data")

    parser.add_option("--trainoutput", help="specify the output location")

    parser.add_option("--testoutput", help="specify the output location")

    parser.add_option("--ratio", help="specify the output location", type="float")

    (options, args) = parser.parse_args()

    # load the data
    main(options.input, options.trainoutput, options.testoutput, options.ratio)