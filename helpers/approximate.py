from optparse import OptionParser
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def main(input_data):

    # read the dataset from file.
    data = pd.read_csv(input_data)

    BertTokenizer.from_pretrained(name, do_lower_case=True)

    # now save the files
    train.to_csv("train.csv", index=None)
    test.to_csv("test.csv", index=None)


if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--input", help="specify the input data")

    (options, args) = parser.parse_args()

    # load the data
    main(options.input)