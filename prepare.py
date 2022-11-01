from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pandas as pd
from helper import parse_data, summarise, encoder, train

pd.options.mode.chained_assignment = None


def main():
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-l", "--label", default="NA", help="File containing labels for training data"
    )
    parser.add_argument("-d", "--data", default="NA", help="Training data")
    parser.add_argument(
        "-m",
        "--model",
        default="SmoteTomek",
        help="Choice of model to used for training (SmoteTomek or BalancedRF)",
    )
    parser.add_argument(
        "-o", "--output", default="model", help="Output name to save trained model"
    )

    args = vars(parser.parse_args())

    label = "./data/" + args["label"]
    data = "./data/" + args["data"]
    model = args["model"]
    output_name = args["output"]

    labels = pd.read_csv(label)
    labels = labels.rename(columns={"transcript_position": "position"})

    gene = parse_data(data)
    full_data = pd.merge(gene, labels, on=["transcript_id", "position"], how="left")
    summarised = summarise(full_data, flag=True)
    encoded = encoder(summarised)
    cols = encoded.columns.tolist()
    new_cols = cols[:1] + [cols[-2]] + [cols[-3]] + [cols[-1]] + cols[1:-3]
    encoded = encoded[new_cols]
    encoded = encoded.astype(
        {"nucleotide-1": "int64", "nucleotide": "int64", "nucleotide+1": "int64"}
    )

    # Saved model will be under results directory
    train(encoded, method=model, out=output_name)


if __name__ == "__main__":
    main()
