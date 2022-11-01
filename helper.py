import os
import pickle

import joblib
import orjson

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder

# from tqdm import tqdm
from xgboost import XGBClassifier

pd.options.mode.chained_assignment = None


def parse_data(data_dir):
    """
    Takes in training data and transform data into new train with the following columns:
        ["transcript_id", "position", "nucleotide", "dwell_1", "std_1", "mean_1",
            "dwell_2", "std_2", "mean_2", "dwell_3", "std_3", "mean_3"]

    Input: training data directory
    ouput: transformed train
    """
    genes = []
    for line in open(data_dir, "r"):
        genes.append(orjson.loads(line))

    lst = []
    for gene in genes:
        transcript_id = next(iter(gene))
        layer = gene[transcript_id]
        position = next(iter(layer))
        next_layer = layer[position]
        nucleotide = next(iter(next_layer))

        rows = [
            [transcript_id, int(position), nucleotide] + i
            for i in next_layer[nucleotide]
        ]
        lst.extend(rows)

    gene = pd.DataFrame(
        lst,
        columns=[
            "transcript_id",
            "position",
            "nucleotide",
            "dwell_1",
            "std_1",
            "mean_1",
            "dwell_2",
            "std_2",
            "mean_2",
            "dwell_3",
            "std_3",
            "mean_3",
        ],
    )
    return gene


def summarise(df, method="mean", flag=False):
    """
    Used to summarise multiple reads of one transcript id into a single data point
    Default method: Mean (Supports median and min-max)

    Input: parsed training data, methods
    Output: Summarised data
    """
    if flag:
        subset_cols = ["gene_id", "transcript_id", "position", "nucleotide"]
        grp_cols = ["gene_id", "transcript_id", "position"]
        on_cols = ["gene_id", "transcript_id", "position", "nucleotide", "label"]
        val_cols = [
            "dwell_1",
            "std_1",
            "mean_1",
            "dwell_2",
            "std_2",
            "mean_2",
            "dwell_3",
            "std_3",
            "mean_3",
            "label",
        ]
    else:
        subset_cols = ["transcript_id", "position", "nucleotide"]
        grp_cols = ["transcript_id", "position"]
        on_cols = ["transcript_id", "position", "nucleotide"]
        val_cols = [
            "dwell_1",
            "std_1",
            "mean_1",
            "dwell_2",
            "std_2",
            "mean_2",
            "dwell_3",
            "std_3",
            "mean_3",
        ]

    nuc = df[subset_cols].groupby(grp_cols)["nucleotide"].unique().reset_index()
    nuc.nucleotide = nuc.nucleotide.apply(lambda x: x[0])

    if method == "mean":
        mean_ds = df.groupby(grp_cols)[val_cols].mean(numeric_only=False).reset_index()
        final_df = mean_ds.merge(nuc)
    elif method == "median":
        median_df = (
            df.groupby(grp_cols)[val_cols].median(numeric_only=False).reset_index()
        )
        final_df = median_df.merge(nuc)
    elif method == "minmax":
        min_df = df.groupby(grp_cols)[val_cols].min(numeric_only=False).reset_index()
        max_df = df.groupby(grp_cols)[val_cols].max(numeric_only=False).reset_index()

        min_df = min_df.merge(nuc)
        max_df = max_df.merge(nuc)

        # rename dataframes
        min_df = min_df.rename(
            columns={
                "dwell_1": "dwell_1_min",
                "std_1": "std_1_min",
                "mean_1": "mean_1_min",
                "dwell_2": "dwell_2_min",
                "std_2": "std_2_min",
                "mean_2": "mean_2_min",
                "dwell_3": "dwell_3_min",
                "std_3": "std_3_min",
                "mean_3": "mean_3_min",
            }
        )
        max_df = max_df.rename(
            columns={
                "dwell_1": "dwell_1_max",
                "std_1": "std_1_max",
                "mean_1": "mean_1_max",
                "dwell_2": "dwell_2_max",
                "std_2": "std_2_max",
                "mean_2": "mean_2_max",
                "dwell_3": "dwell_3_max",
                "std_3": "std_3_max",
                "mean_3": "mean_3_max",
            }
        )

        minmax_data = pd.merge(
            min_df,
            max_df,
            on=on_cols,
            how="left",
        )
        column_to_move = minmax_data.pop("label")
        final_df.insert(22, "label", column_to_move)

    return final_df


def encoder(data, method="train"):
    """
    In the handout, it was explained that the nucleotide column of the dataset represents the
    combined nucleotides from the neighboring 1-flanking position. Since this column is in the form
    of string data, encoding should be carried out to convert these strings into categorical data.

    After the encoding is done, a joblib file would be created to encode the future test set

    Input: Summarised training data, method (either train or test)
    Output: Encoded training data
    """
    if method == "train":
        train = data
        # Here, the nucleotides are split by indexing
        train["nucleotide-1"] = train["nucleotide"].str[0:5]
        train["nucleotide+1"] = train["nucleotide"].str[2:7]
        train["nucleotide"] = train["nucleotide"].str[1:6]

        # Initialise ordinal encoder
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        train[["nucleotide-1", "nucleotide", "nucleotide+1"]] = oe.fit_transform(
            train[["nucleotide-1", "nucleotide", "nucleotide+1"]]
        )

        # Creates a joblib file for future encoding of test set
        if not os.path.exists("./results"):
            os.makedirs("./results")
            joblib.dump(oe, "./results/nucleotide_encoder.joblib")
        else:
            joblib.dump(oe, "./results/nucleotide_encoder.joblib")

        return train

    if method == "test":
        test = data
        oe = joblib.load("./results/nucleotide_encoder.joblib")
        test["nucleotide-1"] = test["nucleotide"].str[0:5]
        test["nucleotide+1"] = test["nucleotide"].str[2:7]
        test["nucleotide"] = test["nucleotide"].str[1:6]

        test[["nucleotide-1", "nucleotide", "nucleotide+1"]] = oe.transform(
            test[["nucleotide-1", "nucleotide", "nucleotide+1"]]
        )

        return test


def smote_tomek_resample(df):
    smt = SMOTETomek(tomek=TomekLinks(sampling_strategy="majority"), random_state=4262)
    X, y = df.drop(columns=["label"]), df["label"]
    X_res, y_res = smt.fit_resample(X, y)
    return X_res, y_res


def prepare_train_test_data(data, train_idx, test_idx, resample_method=False):
    """
    Given indices prepared from test splits, prepare x and y values for train/test from
    initial read data
    - removal of columns is performed within this function
    """
    # Drop identifiers
    if "gene_id" in list(data.columns):
        data = data.drop(columns=["gene_id", "transcript_id", "position"])
    else:
        data = data.drop(columns=["transcript_id", "position"])

    # Split train and test
    train, test = data.iloc[train_idx, :], data.iloc[test_idx, :]

    if not resample_method:
        # Return X_train, y_train, X_test, y_test
        X_train, y_train = train.drop(columns=["label"]), train.label
        X_test, y_test = test.drop(columns=["label"]), test.label
    else:
        # SMOTETomek
        X_train, y_train = smote_tomek_resample(data)
        X_test, y_test = test.drop(columns=["label"]), test.label

    return X_train, y_train, X_test, y_test


def train(df, method="SmoteTomek", out="model"):
    """
    Method used to train model for prediction, allows user to train two types of models
    (SmoteTomek or BalancedRFClassifier)
    Creates a jobib file to save the model when done

    Input: encoded train dataframe, method
    """
    splitter = GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=4262)
    temp = splitter.split(df, groups=df["gene_id"])

    if method.lower() == "smotetomek":
        for _ in range(5):
            train_index, test_index = next(temp)
        X_train, y_train, _, _ = prepare_train_test_data(
            df, train_index, test_index, True
        )
        clf = XGBClassifier(random_state=4262)
    elif method.lower() == "balancedrf":
        for _ in range(2):
            train_index, test_index = next(temp)
        X_train, y_train, _, _ = prepare_train_test_data(
            df, train_index, test_index, False
        )
        clf = BalancedRandomForestClassifier(random_state=4262)

    clf.fit(X_train, y_train)

    filename = "./results/" + out + ".sav"
    pickle.dump(clf, open(filename, "wb"))


def train_val(df, method="SmoteTomek", out="model"):
    """
    Method used to train model for prediction, allows user to train two types of models
    (SmoteTomek or BalancedRFClassifier)
    Purpose: Model training validation

    Input: encoded train dataframe, method
    """
    from sklearn.metrics import (
        auc,
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
    )

    splitter = GroupShuffleSplit(n_splits=5, test_size=0.20, random_state=4262)
    temp = splitter.split(df, groups=df["gene_id"])

    roc, pr, ap = [], [], []
    for train_index, test_index in temp:
        if method.lower() == "smotetomek":
            X_train, y_train, X_test, y_test = prepare_train_test_data(
                df, train_index, test_index, True
            )
            clf = XGBClassifier(random_state=4262)
        elif method.lower() == "balancedrf":
            X_train, y_train, X_test, y_test = prepare_train_test_data(
                df, train_index, test_index, False
            )
            clf = BalancedRandomForestClassifier(random_state=4262)

        # Check distribution of classes
        print(y_train.value_counts())
        print(y_test.value_counts())

        clf.fit(X_train, y_train)

        # For validation:
        test_pred = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()

        print(f"True Negative: {tn}/{tn+fp}")
        print(f"False Positive: {fp}/{tn+fp}")
        print(f"False Negative: {fn}/{fn+tp}")
        print(f"True Positive: {tp}/{fn+tp}")
        roc_auc = roc_auc_score(y_test, test_pred, labels=[0, 1])
        precision_, recall_, _ = precision_recall_curve(y_test, test_pred)
        pr_auc = auc(recall_, precision_)
        aps = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])

        print(f"ROC AUC: {roc_auc}")
        print(f"PR AUC: {pr_auc}")
        print(f"PR AUC #2: {aps}")

        roc.append(roc_auc)
        pr.append(pr_auc)
        ap.append(aps)

    print(f"ROC AUC: {sum(roc)/len(roc)}")
    print(f"PR AUC: {sum(pr)/len(pr)}")
    print(f"PR AUC AVERAGE PRECISION: {sum(ap)/len(ap)}")
