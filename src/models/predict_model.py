""" Module contains functions for training models """

import argparse
import math
import os
import re
import warnings

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
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


### Classes for CustomTransformer Network


class PositionalEncoding(nn.Module):
    """Add positional encoding"""

    def __init__(self, embedding_size: int, dropout: float, max_size: int):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            -torch.arange(0, embedding_size, 2) * math.log(10000) / embedding_size
        )
        pos = torch.arange(0, max_size).reshape(max_size, 1)
        pos_embedding = torch.zeros((max_size, embedding_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        """Make forward pass"""
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    """Learn embedding"""

    def __init__(self, vocab_size: int, embedding_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, tokens: Tensor):
        """Make forward pass"""
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)


class DetoxTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embedding_size: int,
        num_heads: int,
        vocab_size: int,
        feedforward_dim: int,
        max_size: int,
        dropout: float = 0.1,
    ):
        super(DetoxTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(
            embedding_size, dropout=dropout, max_size=max_size
        )
        self.input_embeddings = TokenEmbedding(vocab_size, embedding_size)
        self.output_embeddings = TokenEmbedding(vocab_size, embedding_size)
        self.transformer = Transformer(
            d_model=embedding_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
        )
        self.generator = nn.Linear(embedding_size, vocab_size)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        trg_mask: Tensor,
        src_padding_mask: Tensor,
        trg_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        """Make forward pass"""

        src_embeddings = self.positional_encoding(self.input_embeddings(src))
        trg_embeddings = self.positional_encoding(self.output_embeddings(trg))
        outs = self.transformer(
            src_embeddings,
            trg_embeddings,
            src_mask,
            trg_mask,
            None,
            src_padding_mask,
            trg_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        """Encode data using learned embeddings"""
        return self.transformer.encoder(
            self.positional_encoding(self.input_embeddings(src)), src_mask
        )

    def decode(self, trg: Tensor, memory: Tensor, trg_mask: Tensor):
        """Decode data using learned embeddings"""
        return self.transformer.decoder(
            self.positional_encoding(self.output_embeddings(trg)), memory, trg_mask
        )


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
        model = model.to(self.device)
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
        self.test_df.to_csv(
            os.path.join(self.save_path, "custom_transformer.csv"), index=False
        )


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
        self.device = torch.device(
            "cuda" if self.cuda and torch.cuda.is_available() else "cpu"
        )
        if self.load_path == HUB_LOAD_FLAG:
            self.load_path = "dsomni/pmldl1-bart"
        else:
            self.load_path = os.path.join(self.load_path, "bart")

        self.prefix = "paraphrase following to be nontoxic: \n"

    def predict_and_save(self) -> None:
        """Predict using Bart model and save results"""

        self.logger.log(f"Loading model...\nPath: {self.load_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.load_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.load_path)
        model = model.to(self.device)
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
