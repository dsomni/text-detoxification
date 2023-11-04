""" Module contains functions for results visualization """

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors

### Constants

PREDEFINED_COLORS = [
    "#ff0000",
    "#071efb",
    "#9603fd",
    "#ffd700",
    "#00ff00",
    "#04eafc",
]

RELATIVE_METRICS = ["rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bleu", "wer"]
RELATIVE_METRICS_COLUMNS = ["bart", "custom_transformer"]
ABSOLUTE_METRICS = ["toxicity"]
ABSOLUTE_METRICS_COLUMNS = ["toxic", "nontoxic", "bart", "custom_transformer"]

### Class for Logger instance


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


### Plotting functions


def draw_basic_plots(
    df: pd.DataFrame,
    save_path: str,
    round_value: int = 3,
    figsize=(16, 30),
) -> None:
    """Draw basic insights about metrics

    Args:
        df (pd.DataFrame): data frame with metrics
        save_path (str): path to save final plot
        round_value (int, optional): round precision for metrics. Defaults to 3.
        figsize (tuple, optional): matplotlib figure size. Defaults to (16, 30).
    """
    plt.clf()
    plt.close()

    rows_number = len(RELATIVE_METRICS) + len(ABSOLUTE_METRICS)
    _, axs = plt.subplots(rows_number, 2, figsize=figsize)

    for ax in axs.flat:
        ax.grid()
        ax.set_prop_cycle(color=PREDEFINED_COLORS)

    for i in range(rows_number):
        ax1 = axs[i][0]
        ax2 = axs[i][1]

        if i < len(RELATIVE_METRICS):
            metric = RELATIVE_METRICS[i]
            columns = RELATIVE_METRICS_COLUMNS
        else:
            metric = ABSOLUTE_METRICS[i - len(RELATIVE_METRICS)]
            columns = ABSOLUTE_METRICS_COLUMNS

        ax1.set_title(metric)
        ax2.set_title(f"{metric} | Mean")

        mean_values = []
        for column in columns:
            column_name = f"{column}_{metric}"
            rounded = [round(x, round_value) for x in df[column_name]]
            ax1.scatter(range(len(df)), rounded)
            mean_values.append(df[column_name].mean())

        ax2.bar(columns, mean_values, label=columns, color=PREDEFINED_COLORS)

    plt.tight_layout()
    plt.savefig(save_path)


def draw_difference_plots(
    df: pd.DataFrame,
    save_path: str,
    figsize=(16, 30),
):
    """Draw relative difference in metrics between models

    Args:
        df (pd.DataFrame): data frame with metrics
        save_path (str): path to save final plot
        figsize (tuple, optional): matplotlib figure size. Defaults to (16, 30).
    """
    plt.clf()
    plt.close()

    rows_number = len(RELATIVE_METRICS)
    _, axs = plt.subplots(rows_number, 2, figsize=figsize)

    for ax in axs.flat:
        ax.grid()

    for i in range(rows_number):
        ax1 = axs[i][0]
        ax2 = axs[i][1]

        metric = RELATIVE_METRICS[i]
        columns = RELATIVE_METRICS_COLUMNS

        ax1.set_title(f"{metric} | Difference")
        ax2.set_title(f"{metric} | Percent of maximum")

        differences = df[f"bart_{metric}"] - df[f"custom_transformer_{metric}"]
        ax1.bar(
            range(len(df)),
            differences,
            color=np.where(differences > 0, PREDEFINED_COLORS[0], PREDEFINED_COLORS[1]),
        )
        bart_maximum_percent = (
            sum(df[f"bart_{metric}"] > df[f"custom_transformer_{metric}"]) / len(df) * 100
        )

        ax2.bar(
            columns,
            [bart_maximum_percent, 100 - bart_maximum_percent],
            label=columns,
            color=PREDEFINED_COLORS,
        )

    plt.tight_layout()
    plt.savefig(save_path)


def draw_toxicity_plots(
    df: pd.DataFrame,
    save_path: str,
    n_bins: int = 100,
    eps: float = 5e-3,
    figsize=(16, 10),
):
    """Draw basic insights about toxicity

    Args:
        df (pd.DataFrame): data frame with metrics
        save_path (str): path to save final plot
        n_bins (int, optional): number of bars. Defaults to 100.
        eps (float, optional): the minimum included toxicity. Defaults to 5e-3.
        figsize (tuple, optional): matplotlib figure size. Defaults to (16, 30).
    """
    plt.clf()
    plt.close()

    _, axs = plt.subplots(2, 2, figsize=figsize)

    for ax in axs.flat:
        ax.grid()

    for i, ax in enumerate(axs.flat):
        column = ABSOLUTE_METRICS_COLUMNS[i]
        ax.set_title(f"{column} | Toxicity distribution (> {eps})")
        data = list(filter(lambda x: x > eps, df[f"{column}_toxicity"]))

        values, _, patches = ax.hist(data, bins=n_bins)
        fractions = values / values.max()
        norm = colors.Normalize(fractions.min(), fractions.max())
        for frac, patch in zip(fractions, patches):
            color = plt.cm.viridis(norm(frac))  # type: ignore
            patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(save_path)


def construct_and_save_plots(df: pd.DataFrame, save_path: str, logger: Logger) -> None:
    """Construct and save plots

    Args:
        df (pd.DataFrame): data frame with metrics
        save_path (str): path to save final plot
        logger (Logger): logger instance
    """
    logger.log("Plotting basic insights...")
    draw_basic_plots(df, os.path.join(save_path, "basic.svg"))
    logger.log(f"Successfully saved to {os.path.join(save_path, 'basic.svg')}")

    logger.log("Plotting models difference in metrics...")
    draw_difference_plots(df, os.path.join(save_path, "differences.svg"))
    logger.log(f"Successfully saved to {os.path.join(save_path, 'differences.svg')}")

    logger.log("Plotting toxicity insights...")
    draw_toxicity_plots(df, os.path.join(save_path, "toxicity.svg"))
    logger.log(f"Successfully saved to {os.path.join(save_path, 'toxicity.svg')}")


### Utilities functions


def construct_absolute_path(*relative_path: str) -> str:
    """Turn relative file path to absolute

    Raises:
        FileNotFoundError

    Returns:
        str: absolute path
    """
    absolute_path = os.path.abspath(os.path.join(*relative_path))
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Path {absolute_path} does not exist")
    return absolute_path


def load_metrics_data(path: str, logger: Logger) -> pd.DataFrame:
    """Load metrics data from disk

    Args:
        path (str): path of metrics data
        logger (Logger): logger instance

    Returns:
        pd.DataFrame: metrics data pandas data frame
    """
    metrics_df = pd.read_csv(path)
    logger.log(f"{len(metrics_df)=}")

    return metrics_df


def visualize():
    """Visualize metrics"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Visualize metrics")

    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default="./data/figures",
        help="relative path to save generated plots (default: ./data/figures)",
    )
    parser.add_argument(
        "-l",
        "--load-path",
        type=str,
        dest="load_path",
        default="./data/generated/metrics.csv",
        help="relative path to file with\
            computed metrics (default: ./data/generated/metrics.csv)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )

    namespace = parser.parse_args()
    (
        save_path,
        load_path,
        verbose,
    ) = (
        namespace.save_path,
        namespace.load_path,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)

    # Set up logger
    logger = Logger(verbose)

    # Load data
    load_path = construct_absolute_path(load_path)
    metrics_df = load_metrics_data(load_path, logger)

    # Construct and save plots
    save_path = construct_absolute_path(save_path)
    construct_and_save_plots(metrics_df, save_path, logger)

    logger.log("Done!")


if __name__ == "__main__":
    visualize()
