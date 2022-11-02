from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import dask.dataframe as dd
import pandas as pd
from helper import encoder, parse_data, summarise, train
from tqdm.dask import TqdmCallback

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

    # Parse Data
    gene = parse_data(data)

    # Merge Data with Labels
    sd1 = dd.from_pandas(gene, npartitions=3)
    sd2 = dd.from_pandas(labels, npartitions=3)
    with TqdmCallback(desc="Adding Labels"):
        full_data = sd1.merge(
            sd2, how="left", on=["transcript_id", "position"]
        ).compute()

    # Summarise dataframe, i.e. compress rows
    print("Summarising Data")
    summarised = summarise(full_data, method="mean", flag=True)

    # Encode nucleotides
    print("Encoding Data")
    encoded = encoder(summarised, out = output_name)
    cols = encoded.columns.tolist()
    new_cols = cols[:1] + [cols[-2]] + [cols[-3]] + [cols[-1]] + cols[1:-3]
    encoded = encoded[new_cols]
    encoded = encoded.astype(
        {"nucleotide-1": "int64", "nucleotide": "int64", "nucleotide+1": "int64"}
    )

    # Prepare & Train model
    print("Training model")
    # Saved model will be under results directory
    train(encoded, method=model, out=output_name)


if __name__ == "__main__":
    main()
