from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pickle
from prepare import encoder, parse_data, summarise


def main():
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data", default="NA", help="Test data")
    parser.add_argument(
        "-m", "--model", default="model", help="Saved model name from training"
    )

    args = vars(parser.parse_args())

    data = args["data"]
    filename = args["model"] + ".sav"

    gene = parse_data(data)
    summarised = summarise(gene)
    encoded = encoder(summarised, method="test")

    clf = pickle.load(open(filename, 'rb'))
    test_new = encoded.columns.tolist()[:2] + [encoded.columns.tolist()[-2]] + [encoded.columns.tolist()[-3]] + [encoded.columns.tolist()[-1]] + encoded.columns.tolist()[2:-3]
    encoded = encoded[test_new]
    print(encoded.head())
    test_pred = clf.predict_proba(encoded.drop(columns=["transcript_id", "position"]))[
        :, 1
    ]
    results = encoded[["transcript_id", "position"]]
    results["score"] = test_pred
    #print(test_pred)
    results = results.rename(columns={"position": "transcript_position"})

    filename = data.split(".json")[0] + ".csv"
    results.to_csv(filename, index=False)
    return


if __name__ == "__main__":
    main()
