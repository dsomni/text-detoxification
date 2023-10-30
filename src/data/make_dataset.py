""" Module contains functions for obtaining dataset """

import argparse
import os
import shutil
from zipfile import ZipFile

import pandas as pd
import wget

MANUAL_SEED = 42

DATASET_URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"

# Paths from root directory
INTERIM_DATA_PATH = os.path.join(".", "data/interim")
RAW_DATA_PATH = os.path.join(".", "data/raw")


class Logger:
    """Manage log messages"""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def log(self, message: str):
        """Log message to console

        Args:
            message (str): message to log
        """
        if self.verbose:
            print(message)


""" Collect data functions """


def clear_folder(path: str):
    """Clear folder content and add .gitignore

    Args:
        path (str): path to folder
    """

    # Remove all data from folder
    shutil.rmtree(path)

    # Create empty directory
    os.mkdir(path)

    # Add .gitignore file
    file_path = os.path.join(path, ".gitignore")
    with open(file_path, "a", encoding="utf8"):
        pass


def clear_all_data(paths: list[str], logger: Logger):
    """Clear data in folders

    Args:
        paths (list[str]): folders list
        logger (Logger): logger instance
    """
    logger.log("Clearing folders...")
    for path in paths:
        clear_folder(path)
        logger.log(f"Successfully clear '{os.path.abspath(path)}'")
    logger.log("Folders successfully cleared\n")


def download_dataset_zip(url: str, interim_path: str, logger: Logger) -> str:
    """_summary_

    Args:
        url (str): dataset url
        interim_path (str): path to store intermediate data
        logger (Logger): logger instance

    Returns:
        str: path to downloaded zip
    """
    logger.log("Downloading zip...")
    logger.log(f"URL: {url}")
    zip_path = wget.download(
        url, interim_path, bar=wget.bar_adaptive if logger.verbose else None
    )
    logger.log(f"\nSaved to: '{os.path.abspath(zip_path)}'")
    logger.log("Zip successfully downloaded\n")
    return zip_path


def unpack_dataset(zip_path: str, raw_path: str, logger: Logger):
    """_summary_

    Args:
        zip_path (str): path of downloaded zip file
        raw_path (str): path to store final raw data
        logger (Logger): logger instance
    """
    logger.log("Extracting zip...")
    filenames = []
    with ZipFile(zip_path, "r") as file:
        filenames = file.namelist()
        file.extractall(path=raw_path)
    logger.log(f"\nSaved to: '{os.path.abspath(raw_path)}'")
    logger.log(f"Extracted files: {', '.join(filenames)}")
    logger.log("Zip successfully extracted\n")


def collect_data(dataset_url: str, interim_path: str, raw_path: str, logger: Logger):
    """Collects dataset

    Args:
        dataset_url (str): url of dataset
        interim_path (str): path of interim data
        raw_path (str): path of raw data
        logger (Logger): logger instance
    """
    logger.log("Start collecting data")
    clear_all_data([interim_path, raw_path], logger)
    zip_path = download_dataset_zip(dataset_url, interim_path, logger)
    unpack_dataset(zip_path, raw_path, logger)
    logger.log("Finish collecting data\n")


""" Preprocess data functions """


def remove_almost_same_data(
    df: pd.DataFrame,
    similarity_threshold: float = 0.94,
    length_diff_threshold: float = 0.02,
) -> pd.DataFrame:
    """Remove data with high similarity

    Args:
        df (pd.DataFrame): initial dataset
        similarity_threshold (float, optional): Defaults to 0.94.
        length_diff_threshold (float, optional): Defaults to 0.02.

    Returns:
        pd.DataFrame: dataset with no similar data
    """
    return df[
        (df["similarity"] < similarity_threshold)
        & (df["lenght_diff"] > length_diff_threshold)
    ]


def extract_relevant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract initially relevant data and rename columns

    Args:
        df (pd.DataFrame): initial dataset

    Returns:
        pd.DataFrame: dataset with 'toxic' and 'nontoxic' columns
    """
    relevant_data = df[df["ref_tox"] > df["trn_tox"]]
    relevant_data = relevant_data[["reference", "translation", "ref_tox", "trn_tox"]]
    return relevant_data.rename(columns={"reference": "toxic", "translation": "nontoxic"})


def extract_irrelevant_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract initially irrelevant data and rename columns

    Args:
        df (pd.DataFrame): initial dataset

    Returns:
        pd.DataFrame: dataset with swapped 'toxic' and 'nontoxic' columns
    """
    irrelevant_data = df[df["ref_tox"] <= df["trn_tox"]]
    irrelevant_data = irrelevant_data[["reference", "translation", "ref_tox", "trn_tox"]]
    return irrelevant_data.rename(
        columns={"reference": "nontoxic", "translation": "toxic"}
    )


def build_relevant_dataset(df: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    """Create main dataset

    Args:
        df (pd.DataFrame): initial dataset
        logger (Logger): logger instance

    Returns:
        pd.DataFrame: clean dataset with 'toxic' and 'nontoxic' columns
    """
    logger.log("Cleaning dataset...")
    clean_df = remove_almost_same_data(df)
    logger.log("Extracting relevant data...")
    relevant_data = extract_relevant_data(clean_df)
    logger.log("Extracting irrelevant data...")
    irrelevant_data = extract_irrelevant_data(clean_df)

    return pd.concat([relevant_data, irrelevant_data])


def retain_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only "toxic" and "nontoxic" columns

    Returns:
        pd.DataFrame
    """
    return df[["toxic", "nontoxic"]]


def retain_representative_data(
    df: pd.DataFrame, toxicity_threshold: float = 0.9, no_toxicity_threshold: float = 0.1
) -> pd.DataFrame:
    """Retain only high representative data

    Args:
        df (pd.DataFrame): initial dataset
        toxicity_threshold (float, optional): Defaults to 0.9.
        no_toxicity_threshold (float, optional): Defaults to 0.05.

    Returns:
        pd.DataFrame: representative dataset
    """
    return df[
        (df["ref_tox"] >= toxicity_threshold) & (df["trn_tox"] <= no_toxicity_threshold)
    ]


def build_different_sizes(
    df: pd.DataFrame, logger: Logger
) -> list[tuple[str, pd.DataFrame]]:
    """Build dataset of different sizes from initial

    Args:
        df (pd.DataFrame): initial dataset
        logger (Logger): logger instance

    Returns:
        list[tuple[str, pd.DataFrame]]: List of (name, dataset) pairs
    """
    size_map = {
        "lg": {"toxicity_threshold": 0.9, "no_toxicity_threshold": 0.1},
        "md": {"toxicity_threshold": 0.99, "no_toxicity_threshold": 0.01},
        "sm": {"toxicity_threshold": 0.999, "no_toxicity_threshold": 0.001},
        "xs": {"toxicity_threshold": 0.9994, "no_toxicity_threshold": 0.0001},
    }

    datasets = []
    for size, args in size_map.items():
        name = f"dataset_{size}"
        datasets.append((name, retain_representative_data(df, **args)))
        logger.log(f"Finish with '{name}'")
    return datasets


def build_train_dataset(
    df: pd.DataFrame, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build test dataset from test_size

    Args:
        df (pd.DataFrame): initial dataset
        test_size (int): size of test dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: test dataset and
        initial dataset without test data
    """

    frac = test_size / len(df)

    test_df = df.sample(frac=frac, random_state=MANUAL_SEED)

    rest_df = df.drop(test_df.index)

    return test_df, rest_df


def load_dataset(path: str, **kwargs) -> pd.DataFrame:
    """Read csv or tsv file from disk

    Args:
        path (str)

    Returns:
        pd.DataFrame
    """
    return pd.read_csv(path, **kwargs)


def save_dataset(df: pd.DataFrame, path: str, **kwargs) -> None:
    """Save dataset to disk

    Args:
        df (pd.DataFrame)
        path (str)
    """
    df.to_csv(path, **kwargs)


def preprocess_dataset(
    raw_dataset_path: str, test_size: int, logger: Logger
) -> list[tuple[str, pd.DataFrame]]:
    """Collects dataset

    Args:
        raw_dataset_path (str): path of row dataset
        test_size (int): size of test dataset
        logger (Logger): logger instance

    Returns:
        list[tuple[str, pd.DataFrame]]: preprocessed datasets
        in the form (name, dataset)
    """
    logger.log("Start preprocessing data")
    raw_df = load_dataset(raw_dataset_path, delimiter="\t")

    logger.log("\nBuilding relevant dataset")
    rel_df = build_relevant_dataset(raw_df, logger)

    logger.log("\nBuilding representative dataset")
    rep_df = retain_representative_data(rel_df)

    logger.log("\nBuilding test dataset")
    test_df, rest_df = build_train_dataset(rep_df, test_size)

    datasets: list[tuple[str, pd.DataFrame]] = [("test", test_df)]

    logger.log("\nConstructing different datasets")
    datasets.extend(build_different_sizes(rest_df, logger))

    logger.log("\nFinish preprocessing data\n")
    return [(name, retain_useful_columns(df)) for (name, df) in datasets]


def make_dataset():
    """Make dataset"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Make dataset script")
    parser.add_argument(
        "-i",
        "--interim-path",
        type=str,
        dest="interim",
        default=INTERIM_DATA_PATH,
        help="path to save the intermediate data like .zip files",
    )
    parser.add_argument(
        "-r",
        "--raw-path",
        type=str,
        dest="raw",
        default=RAW_DATA_PATH,
        help="path to save the raw data like .tsv files",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        dest="url",
        default=DATASET_URL,
        help="url to download data from",
    )
    parser.add_argument(
        "-t",
        "--test-size",
        type=int,
        dest="test_size",
        default=500,
        help="size of test dataset (default: 500)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )
    namespace = parser.parse_args()
    test_size, url, interim_path, raw_path, verbose = (
        namespace.test_size,
        namespace.url,
        namespace.interim,
        namespace.raw,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)

    # Set up logger
    logger = Logger(verbose)

    # Collecting data
    collect_data(url, interim_path, raw_path, logger)

    # Preprocess dataset
    raw_filename = "filtered.tsv"
    datasets = preprocess_dataset(os.path.join(raw_path, raw_filename), test_size, logger)

    logger.log("Saving datasets...\n")
    for name, dataset in datasets:
        save_dataset(dataset, os.path.join(raw_path, f"{name}.csv"), index=False)

    logger.log("Done!")


if __name__ == "__main__":
    make_dataset()
