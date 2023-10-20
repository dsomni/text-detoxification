""" Module contains functions for obtaining dataset """

import argparse
import os
import shutil
from zipfile import ZipFile

import pandas as pd
import wget

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
    return df[
        (df["similarity"] < similarity_threshold)
        & (df["lenght_diff"] > length_diff_threshold)
    ]


def extract_relevant_data(df: pd.DataFrame) -> pd.DataFrame:
    relevant_data = df[df["ref_tox"] > df["trn_tox"]]
    relevant_data = relevant_data[["reference", "translation"]]
    return relevant_data.rename(columns={"reference": "toxic", "translation": "nontoxic"})


def extract_irrelevant_data(df: pd.DataFrame) -> pd.DataFrame:
    irrelevant_data = df[df["ref_tox"] <= df["trn_tox"]]
    irrelevant_data = irrelevant_data[["reference", "translation"]]
    return irrelevant_data.rename(
        columns={"reference": "nontoxic", "translation": "toxic"}
    )


def build_dataset(df: pd.DataFrame, logger: Logger) -> pd.DataFrame:
    logger.log("Cleaning dataset...")
    clean_df = remove_almost_same_data(df)
    logger.log("Extracting relevant data...")
    relevant_data = extract_relevant_data(clean_df)
    logger.log("Extracting irrelevant data...")
    irrelevant_data = extract_irrelevant_data(clean_df)

    return pd.concat([relevant_data, irrelevant_data])


def load_dataset(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_dataset(df: pd.DataFrame, path: str, **kwargs) -> None:
    df.to_csv(path, **kwargs)


def preprocess_dataset(raw_dataset_path: str, save_path: str, logger: Logger):
    """Collects dataset

    Args:
        raw_dataset_path (str): path of row dataset
        save_path (str): path to save preprocessed dataset
        logger (Logger): logger instance
    """
    logger.log("Start preprocessing data")
    raw_df = load_dataset(raw_dataset_path, delimiter="\t")
    preprocessed_df = build_dataset(raw_df, logger)
    save_dataset(preprocessed_df, save_path, index=False)

    logger.log("Finish preprocessing data\n")


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
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )
    namespace = parser.parse_args()
    url, interim_path, raw_path, verbose = (
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
    preprocessed_filename = "dataset.csv"
    preprocess_dataset(
        os.path.join(raw_path, raw_filename),
        os.path.join(raw_path, preprocessed_filename),
        logger,
    )

    logger.log("Done!")


if __name__ == "__main__":
    make_dataset()
