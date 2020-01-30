import glob, os, re
import pandas as pd
import seaborn as sns;
from optparse import OptionParser

sns.set()
import matplotlib.pyplot as plt


def process_data(expname, split):
    results = None
    for filename in glob.iglob("/Users/apatterson/Desktop/" + expname + "/" + "**/" + split + "/summary.csv",
                               recursive=True):
        # extract the epoch number
        df = pd.read_csv(filename, index_col=0)
        df['EPOCH'] = int(filename.split("/")[-3:-2][0])

        if results is not None:
            results = pd.concat([results, df], axis=0)

        else:
            results = df
    results['data_split'] = split
    return results

if __name__ == "__main__":

    parser = OptionParser()

    parser.add_option("--input", help="specify the input data")

    exp = "test3"
    train = process_data(exp, "train")
    validation = process_data(exp, "validation")
    results = pd.concat([train, validation], axis=0)
    ax = sns.lineplot(x="EPOCH", y="LOSS", data=results, hue="data_split")
    plt.show()
