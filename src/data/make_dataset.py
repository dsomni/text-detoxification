""" Module contains functions for obtaining dataset """

import argparse
import os
import shutil
from zipfile import ZipFile

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
    clear_all_data([interim_path, raw_path], logger)
    zip_path = download_dataset_zip(url, interim_path, logger)
    unpack_dataset(zip_path, raw_path, logger)

    logger.log("Done!")


if __name__ == "__main__":
    make_dataset()
