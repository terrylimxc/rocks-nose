import pickle
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from prepare import encoder, parse_data, summarise


def main():
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data", default="NA", help="Test data")
    parser.add_argument(
        "-m", "--model", default="model", help="Saved model name from training"
    )

    args = vars(parser.parse_args())

    data = "./data/" + args["data"]
    model = args["model"]
    filename = "./results/" + args["model"] + ".sav"

    # Parse Data
    gene = parse_data(data)

    # Summarise dataframe, i.e. compress rows
    print("Summarising Data")
    summarised = summarise(gene)

    # Encode nucleotides
    print("Encoding Data")
    encoded = encoder(summarised, method="test", out=model)
    encoded_filename = (data.split(".json")[0]).split("./data/")[-1]
    encoded.to_csv(f"./results/encoded_{encoded_filename}.csv", index=False)

    # # Prepare and test model
    # print("Running Model")
    # clf = pickle.load(open(filename, "rb"))
    # cols = encoded.columns.tolist()
    # test_new = cols[:2] + [cols[-2]] + [cols[-3]] + [cols[-1]] + cols[2:-3]
    # encoded = encoded[test_new]

    # test_pred = clf.predict_proba(encoded.drop(columns=["transcript_id", "position"]))[
    #     :, 1
    # ]
    # results = encoded[["transcript_id", "position"]]
    # results["score"] = test_pred
    # results = results.rename(columns={"position": "transcript_position"})

    # print("Saving Results")
    # filename = "./results/" + (data.split(".json")[0]).split("./data/")[-1] + ".csv"
    # results.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
