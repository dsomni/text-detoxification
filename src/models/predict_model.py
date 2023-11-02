""" Module contains functions for training models """

import argparse
import os
import re
import warnings

import pandas as pd
import torch
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

AVAILABLE_MODELS = ["bart", "custom_transformer"]
HUB_LOAD_FLAG = "HUB"


### Classes for CustomTransformer Network


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


### Tester Classes


class CustomTransformerTester:
    """Custom Transformer tester for Text Detoxification"""

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones((size, size), device=self.device)) == 1).transpose(
            0, 1
        )
        return (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )

    def _preprocess_text(self, text: str, vocab, tokenizer) -> torch.Tensor:
        return torch.tensor([self.bos_idx, *vocab(tokenizer(text.lower())), self.eos_idx])

    def _decode_tokens(self, tokens: torch.Tensor, vocab) -> str:
        text = (
            " ".join(vocab.lookup_tokens(list(tokens.cpu().numpy())))
            .replace("<bos>", "")
            .replace("<eos>", "")
            .strip()
        )
        return re.sub(" +", " ", re.sub(r'\s([?.!"](?:\s|$))', r"\1", text))

    def _greedy_decode(
        self,
        model: torch.nn.Module,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        max_size: int,
        start_symbol: int,
    ) -> torch.Tensor:
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = model.encode(src, src_mask)
        answer = torch.ones(1, 1).fill_(start_symbol).long().to(self.device)
        for _ in range(max_size - 1):
            memory = memory.to(self.device)

            trg_mask = (self._generate_square_subsequent_mask(answer.size(0)).bool()).to(
                self.device
            )
            outputs = model.decode(answer, memory, trg_mask)
            outputs = outputs.transpose(0, 1)

            probabilities = model.generator(outputs[:, -1])
            _, next_word = torch.max(probabilities, dim=1)
            next_word = next_word.item()

            answer = torch.cat(
                [answer, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0
            )
            if next_word == self.eos_idx:
                break
        return answer

    def _detoxify(
        self, model: torch.nn.Module, src_sentence: str, vocab, tokenizer
    ) -> str:
        src = self._preprocess_text(src_sentence, vocab, tokenizer).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        output_tokens = self._greedy_decode(
            model, src, src_mask, max_size=num_tokens + 5, start_symbol=self.bos_idx
        ).flatten()
        return self._decode_tokens(output_tokens, vocab)

    def __init__(
        self,
        test_df: pd.DataFrame,
        load_path: str,
        cuda: bool,
        save_path: str,
        logger: Logger,
    ) -> None:
        self.test_df = test_df
        self.load_path = load_path
        self.cuda = cuda
        self.save_path = save_path
        self.logger = logger

        # Constants for the training process
        self.bos_idx, self.eos_idx, self.pad_idx = 1, 2, 3
        self.device = torch.device(
            "cuda" if self.cuda and torch.cuda.is_available() else "cpu"
        )

    def predict_and_save(self) -> None:
        """Predict using Custom Transformer model and save results"""

        self.logger.log("Loading vocab...")
        vocab = torch.load(os.path.join(self.load_path, "custom_transformer_vocab.pth"))

        self.logger.log(f"Loading model...\nPath: {self.load_path}")
        model = torch.load(os.path.join(self.load_path, "custom_transformer"))
        model.eval()

        self.logger.log("Loading tokenizer...")
        tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
        self.bos_idx, self.eos_idx, self.pad_idx = vocab(["<bos>", "<eos>", "<pad>"])

        self.logger.log("Predicting...")
        if self.logger.verbose:
            loop = tqdm(self.test_df.iterrows(), total=len(self.test_df))
        else:
            loop = self.test_df.iterrows()
        model_answers = []
        for _, r in loop:
            model_answers.append(self._detoxify(model, r["toxic"], vocab, tokenizer))
        self.test_df["generated"] = model_answers

        self.logger.log(f"Saving results...\nPath: {self.save_path}")
        self.test_df.to_csv(os.path.join(self.save_path, "bart.csv"), index=False)


class BartTester:
    """Tester class for fine-tunned BART for Text Detoxification task"""

    def _detoxify(self, model, tokenizer, prompt: str) -> str:
        inference_request = self.prefix + prompt
        input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
        outputs = model.generate(input_ids=input_ids)
        return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)

    def __init__(
        self,
        test_df: pd.DataFrame,
        load_path: str,
        cuda: bool,
        save_path: str,
        logger: Logger,
    ) -> None:
        self.test_df = test_df
        self.load_path = load_path
        self.cuda = cuda
        self.save_path = save_path
        self.logger = logger

        # Constants for the training process
        if self.load_path == HUB_LOAD_FLAG:
            self.load_path = "dsomni/pmldl1-bart"

        self.prefix = "paraphrase following to be nontoxic: \n"

    def predict_and_save(self) -> None:
        """Predict using Bart model and save results"""

        self.logger.log(f"Loading model...\nPath: {self.load_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.load_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.load_path)
        model.eval()
        model.config.use_cache = False

        self.logger.log("Predicting...")
        if self.logger.verbose:
            loop = tqdm(self.test_df.iterrows(), total=len(self.test_df))
        else:
            loop = self.test_df.iterrows()
        model_answers = []
        for _, r in loop:
            model_answers.append(self._detoxify(model, tokenizer, r["toxic"]))
        self.test_df["generated"] = model_answers

        self.logger.log(f"Saving results...\nPath: {self.save_path}")
        self.test_df.to_csv(os.path.join(self.save_path, "bart.csv"), index=False)


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


def load_test_data(test_path: str, logger: Logger) -> pd.DataFrame:
    """Load test data from disk

    Args:
        train_path (str): path of training dataset
        logger (Logger): logger instance

    Returns:
        pd.DataFrame: training data pandas data frame
    """
    test_df = pd.read_csv(test_path)
    logger.log(f"{len(test_df)=}")

    return test_df


def predict_model():
    """Train and save model"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and save model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model",
        choices=AVAILABLE_MODELS,
        default="bart",
        help="model to train (default: bart)",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default="./data/generated",
        help="relative path to save generated results (default: ./data/generated)",
    )
    parser.add_argument(
        "-d",
        "--data-load-path",
        type=str,
        dest="data_load_path",
        default="./data/raw/test.csv",
        help="relative path to load test data from (default: ./data/raw/test.csv)",
    )
    parser.add_argument(
        "-l",
        "--load-path",
        type=str,
        dest="load_path",
        default="./models",
        help=f"relative path to load models from. \
          set '{HUB_LOAD_FLAG}' to load HuggingFace \
              model from HuggingFace Hub. (default: ./models)",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="use CUDA if available (default: True)",
    )
    parser.add_argument(
        "-i",
        "--ignore-warnings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="ignore warnings (default: True)",
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
        model,
        save_path,
        data_load_path,
        load_path,
        cuda,
        ignore_warnings,
        verbose,
    ) = (
        namespace.model,
        namespace.save_path,
        namespace.data_load_path,
        namespace.load_path,
        namespace.cuda,
        namespace.ignore_warnings,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)
    cuda: bool = bool(cuda)
    ignore_warnings: bool = bool(ignore_warnings)

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    save_path = construct_absolute_path(save_path)

    # Set up logger
    logger = Logger(verbose)

    # Load datasets
    load_test_path = construct_absolute_path(data_load_path)

    test_df = load_test_data(load_test_path, logger)

    if model != "bart" or load_path != HUB_LOAD_FLAG:
        load_path = construct_absolute_path(load_path)

    # Test model
    if model == "bart":
        tester = BartTester(test_df, load_path, cuda, save_path, logger)
    else:  # elif model == "custom_transformer"
        tester = CustomTransformerTester(test_df, load_path, cuda, save_path, logger)

    tester.predict_and_save()

    logger.log("Done!")


if __name__ == "__main__":
    predict_model()
