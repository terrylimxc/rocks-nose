from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import joblib
from prepare import encoder, parse_data, summarise



def main():
    """
    TO-DO

    Add in predictions for unseen test set
    """
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data", default="NA", help="Test data")
    parser.add_argument(
        "-m", "--model", default="model", help="Saved model name from training"
    )

    args = vars(parser.parse_args())

    data = args["data"]
    saved_model = args["model"] + ".joblib"

    gene = parse_data(data)
    summarised = summarise(gene)
    encoded = encoder(summarised, method="test")

    clf = joblib.load(saved_model)
    test_pred = clf.predict_proba(encoded.drop(columns=["transcript_id", "position"]))[
        :, 1
    ]

    results = encoded[["transcript_id", "position"]]
    results["score"] = test_pred
    results = results.rename(columns={"position": "transcript_position"})

    filename = data + ".csv"
    joblib.dump(results, filename)
    return


if __name__ == "__main__":
    main()
