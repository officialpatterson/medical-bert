import sys
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def resample_data(t):
    t = t[['text', 'readm_30d']]
    label = t.pop('readm_30d')

    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(t, label.astype('category'))

    df = pd.DataFrame(X[:, 0])
    df.columns = ['text']
    df['readm_30d'] = pd.Series(y)

    return df


if __name__ == '__main__':
    input_file = args = sys.argv[1]
    output_file = args = sys.argv[2]

    df = pd.read_csv(input_file)

    df = resample_data(df)

    print("new shape: {}".format(df.shape))

    df.to_csv(output_file)
