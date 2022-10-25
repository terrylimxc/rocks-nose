# import json
# from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# from code import encoder, parse_data, summarise
# import joblib


def main():
    """
    TO-DO

    Add in predictions for unseen test set

    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data", default="NA", help="Test data")
    parser.add_argument(
        "-m",
        "--model",
        default="model",
        help="Saved model name from training",
    )

    args = vars(parser.parse_args())

    data = args["data"]
    saved_model = args["model"] + ".joblib"

    gene = parse_data(data)
    summarised = summarise(gene)
    encoded = encoder(summarised, method="test")

    clf = joblib.load(saved_model)
    """

    return


if __name__ == "__main__":
    main()
