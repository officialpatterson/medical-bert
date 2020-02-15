import glob, os, re
import pandas as pd
import seaborn as sns;
from optparse import OptionParser

sns.set()
import matplotlib.pyplot as plt


def process_data(experimentPath, split):
    results = None
    path = experimentPath + "/" + "**/" + split + "/summary.csv"
    for filename in glob.iglob(path, recursive=True):
        # extract the epoch number
        df = pd.read_csv(filename, index_col=0)
        df['EPOCH'] = int(filename.split("/")[-3:-2][0])

        if results is not None:
            results = pd.concat([results, df], axis=0)

        else:
            results = df
    results['data_split'] = split
    results['expname'] = experimentPath
    return results

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--input", help="specify the input data")

    parser.add_option("--y_columns", help="y column name")
    parser.add_option("--save_dataframe", help="location to save the data frame to ")
    (options, args) = parser.parse_args()

    train = process_data(options.input, "train")
    validation = process_data(options.input, "validation")
    results = pd.concat([train, validation], axis=1)


    results.to_csv(options.save_dataframe)
    ax = sns.lineplot(x="EPOCH", y="LOSS", data=results['train'])
    plt.show()
