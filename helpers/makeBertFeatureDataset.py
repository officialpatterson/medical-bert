def main(input_data, output_dir, ratio):

    # read the dataset from file.
    print("Reading raw data")
    data = pd.read_csv(input_data)

    # split into training and testing
    print("Splitting into training and testing")
    train, test = split_data(data, ratio)

    # split into train and validation
    print("spliting train into train and validation")
    train, validation = split_data(train, ratio)

    # undersample the train
    print("Undersampling the train")
    train = resample_data(train)

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