import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from optparse import OptionParser
from sklearn.model_selection import train_test_split

#This will make a train/validation/test split 80/20/20
def resample_data(t):
    t = t[['HADM_ID', 'text', 'readm_30d']]
    label = t.pop('readm_30d')

    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(t, label.astype('category'))

    ids = pd.Series(X[:, 0])
    texts = pd.Series(X[:, 1])

    df = pd.DataFrame()
    df['readm_30d'] = pd.Series(y)
    df['HADM_ID'] = ids
    df['text'] = texts
    return df

def split_data(admissions, ratio):

    # Do some limited preprocessing
    X = admissions[['HADM_ID', 'text']]
    y = admissions['readm_30d']

    # Create a stratified train test split to preserver distribution.
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=ratio, random_state=42)

    train = pd.merge(X_train, y_train, left_index=True, right_index=True)
    test = pd.merge(X_test, y_test, left_index=True, right_index=True)

    return train, test


def main(input_data, output_dir, ratio):

    # read the dataset from file.
    print("Reading raw data")
    data = pd.read_csv(input_data)

    # split into training and testing
    print("Splitting into training and testing")
    train, test = split_data(data, ratio)

    # undersample the train
    print("Undersampling the train")
    train = resample_data(train)

    # split into train and validation
    print("spliting train into train and validation")
    train, validation = split_data(train, ratio)

    # now save the files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train.to_csv(os.path.join(output_dir, "train.csv"), index=None)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=None)
    validation.to_csv(os.path.join(output_dir, "validation.csv"), index=None)

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--input", help="specify the input data")

    parser.add_option("--output_dir", help="specify the output location")

    parser.add_option("--ratio", help="specify the proportion to keep for testing", type="float")

    (options, args) = parser.parse_args()

    # load the data
    main(options.input, options.output_dir, options.ratio)